#!/usr/bin/env python3
"""
Neo4j Aura Knowledge Graph Loader
==================================

Loads the Persona Solid Cancer Knowledge Graph (KG) JSON into Neo4j Aura.

Features:
- Schema setup: Unique constraints on node IDs and indexes on names, labels, domains.
- Lossless node import with primary semantic labels (Role, Domain, Topic, etc.) and KGNode.
- Lossless relationship import with deterministic edge IDs for idempotency.
- Fast, batched Cypher executions.
- Optional --reset flag to clear existing graph data before importing.

Usage:
    python neo4j_aura_kg.py
    python neo4j_aura_kg.py --reset
    python neo4j_aura_kg.py --json "path/to/Persona_Solid_Cancer.json"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

DEFAULT_JSON = Path(__file__).with_name("Persona_Solid_Cancer_OS_FRM.json")

ALLOWED_LABELS = {
    "Role",
    "Domain",
    "Topic",
    "CoreResponsibility",
    "ExclusionRule",
    "DocumentationRule",
    "CollaborationRule",
    "ComplianceRule",
    "Account",
    "AccountBarrier",
}

ALLOWED_EDGE_TYPES = {
    "OPERATES_IN",
    "HAS_RESPONSIBILITY",
    "HAS_EXCLUSION_RULE",
    "HAS_DOCUMENTATION_RULE",
    "HAS_COLLABORATION_RULE",
    "ABOUT_TOPIC",
    "EXCLUDES_TOPIC",
    "DOCUMENTS_TOPIC",
    "RELATES_TO_TOPIC",
    "HAS_IN_SCOPE_TOPIC",
    "HAS_OUT_OF_SCOPE_TOPIC",
    "COLLABORATES_WITH",
    "GOVERNED_BY",
    "ENGAGES_ACCOUNT",
    "HAS_ACCOUNT_BARRIER",
    "GOVERNS_ROLE",
}

SAFE_REL_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def die(msg: str) -> None:
    print(f"\n[FATAL ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def load_kg(path: Path) -> dict[str, Any]:
    """Loads and validates the Knowledge Graph JSON file."""
    if not path.exists():
        die(f"KG JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        kg = json.load(f)

    if not isinstance(kg, dict):
        die("KG root must be a JSON object.")
    if not isinstance(kg.get("nodes"), list):
        die("KG must contain a 'nodes' array.")
    if not isinstance(kg.get("edges"), list):
        die("KG must contain an 'edges' array.")

    return kg


def connect():
    """Initializes the Neo4j GraphDatabase driver using environment variables."""
    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not uri or not user or not password:
        die("Missing required environment variables in .env (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD).")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    return driver, database


def run_query(driver, database: str, cypher: str, **params):
    """Executes a Cypher query against the specified Neo4j database."""
    records, summary, keys = driver.execute_query(
        cypher,
        parameters_=params,
        database_=database,
    )
    return records


def reset_db(driver, database: str):
    """Deletes all existing nodes and relationships from the database."""
    print("Resetting database: clearing all existing nodes and relationships...")
    run_query(
        driver,
        database,
        """
        MATCH (n)
        DETACH DELETE n
        """,
    )


def create_schema(driver, database: str):
    """Creates constraints and indexes for schema integrity and performance."""
    print("Configuring schema constraints and indexes...")
    queries = [
        """
        CREATE CONSTRAINT kg_node_id_unique IF NOT EXISTS
        FOR (n:KGNode)
        REQUIRE n.id IS UNIQUE
        """,
        """
        CREATE INDEX kg_node_name IF NOT EXISTS
        FOR (n:KGNode) ON (n.name)
        """,
        """
        CREATE INDEX kg_node_label IF NOT EXISTS
        FOR (n:KGNode) ON (n.kg_label)
        """,
        """
        CREATE INDEX kg_node_domain IF NOT EXISTS
        FOR (n:KGNode) ON (n.domain)
        """,
    ]

    for q in queries:
        run_query(driver, database, q)


def edge_id(source: str, target: str, rel_type: str, properties: dict[str, Any]) -> str:
    """Generates a deterministic unique ID for each relationship."""
    payload = json.dumps(
        {
            "source": source,
            "target": target,
            "type": rel_type,
            "properties": properties,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    return f"edge:{digest}"


def import_kg(driver, database: str, kg: dict[str, Any]):
    """Imports nodes and edges into Neo4j using efficient batch queries."""
    nodes = kg["nodes"]
    edges = kg["edges"]

    print(f"Importing {len(nodes)} nodes across {len(ALLOWED_LABELS)} label types...")

    # Group nodes by primary label for single-roundtrip batch ingestion
    nodes_by_label = defaultdict(list)
    for n in nodes:
        label = n.get("label")
        if label not in ALLOWED_LABELS:
            raise ValueError(f"Unsupported label: {label}")

        props = dict(n.get("properties") or {})
        list_props = {k: v for k, v in props.items() if isinstance(v, list)}
        scalar_props = {
            k: v
            for k, v in props.items()
            if not isinstance(v, list) and (isinstance(v, (str, int, float, bool)) or v is None)
        }

        nodes_by_label[label].append(
            {
                "id": n["id"],
                "properties_json": json.dumps(props, ensure_ascii=False, sort_keys=True),
                "props": scalar_props,
                "list_props": list_props,
            }
        )

    for label, batch in nodes_by_label.items():
        cypher = f"""
        UNWIND $batch AS item
        MERGE (n:KGNode:{label} {{id: item.id}})
        SET n.kg_label = '{label}',
            n.properties_json = item.properties_json,
            n += item.props,
            n += item.list_props
        """
        run_query(driver, database, cypher, batch=batch)

    print(f"Importing {len(edges)} relationships across {len(ALLOWED_EDGE_TYPES)} types...")

    # Group edges by type for single-roundtrip batch ingestion
    edges_by_type = defaultdict(list)
    for e in edges:
        rel_type = e.get("type")
        if rel_type not in ALLOWED_EDGE_TYPES or not SAFE_REL_RE.match(rel_type):
            raise ValueError(f"Unsupported or unsafe relationship type: {rel_type}")

        props = dict(e.get("properties") or {})
        eid = edge_id(e["source"], e["target"], rel_type, props)
        scalar_props = {
            k: v for k, v in props.items() if isinstance(v, (str, int, float, bool)) or v is None
        }

        edges_by_type[rel_type].append(
            {
                "source": e["source"],
                "target": e["target"],
                "edge_id": eid,
                "properties_json": json.dumps(props, ensure_ascii=False, sort_keys=True),
                "props": scalar_props,
            }
        )

    for rel_type, batch in edges_by_type.items():
        cypher = f"""
        UNWIND $batch AS item
        MATCH (a:KGNode {{id: item.source}})
        MATCH (b:KGNode {{id: item.target}})
        MERGE (a)-[r:{rel_type} {{edge_id: item.edge_id}}]->(b)
        SET r.properties_json = item.properties_json,
            r += item.props
        """
        run_query(driver, database, cypher, batch=batch)


def print_graph_summary(driver, database: str):
    """Fetches and displays a concise summary of the loaded graph."""
    counts_rec = run_query(
        driver,
        database,
        """
        MATCH (n:KGNode)
        WITH count(n) AS total_nodes
        MATCH ()-[r]->()
        RETURN total_nodes, count(r) AS total_edges
        """,
    )
    total_nodes = counts_rec[0]["total_nodes"] if counts_rec else 0
    total_edges = counts_rec[0]["total_edges"] if counts_rec else 0

    labels_rec = run_query(
        driver,
        database,
        """
        MATCH (n:KGNode)
        RETURN n.kg_label AS label, count(n) AS count
        ORDER BY count DESC
        """,
    )

    roles_rec = run_query(
        driver,
        database,
        """
        MATCH (r:Role)-[:OPERATES_IN]->(d:Domain)
        OPTIONAL MATCH (r)-[:HAS_RESPONSIBILITY]->(resp:CoreResponsibility)
        OPTIONAL MATCH (r)-[:HAS_EXCLUSION_RULE]->(excl:ExclusionRule)
        RETURN r.name AS role,
               r.full_name AS full_name,
               d.name AS domain,
               count(DISTINCT resp) AS responsibilities,
               count(DISTINCT excl) AS exclusions
        ORDER BY r.name
        """,
    )

    print("\n" + "=" * 70)
    print("             NEO4J KNOWLEDGE GRAPH SUMMARY             ")
    print("=" * 70)
    print(f"Database:        {database}")
    print(f"Total Nodes:     {total_nodes}")
    print(f"Total Edges:     {total_edges}")
    print("-" * 70)
    print("Node Labels:")
    for row in labels_rec:
        print(f"  - {row['label']:<24}: {row['count']:>4}")
    print("-" * 70)
    print("Roles & Personas:")
    for row in roles_rec:
        print(f"  - [{row['role']:<4}] {row['full_name']} ({row['domain']})")
        print(f"         Responsibilities: {row['responsibilities']}, Exclusions: {row['exclusions']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Load Persona Solid Cancer Knowledge Graph into Neo4j Aura.")
    parser.add_argument("--json", default=str(DEFAULT_JSON), help="Path to KG JSON file (default: Persona_Solid_Cancer_OS_FRM.json)")
    parser.add_argument("--reset", action="store_true", help="Delete existing graph nodes/edges before import")
    args = parser.parse_args()

    json_path = Path(args.json).resolve()
    print(f"Loading Knowledge Graph data from: {json_path.name}")
    kg = load_kg(json_path)

    driver = None
    try:
        t0 = time.time()
        driver, database = connect()
        print(f"Successfully connected to Neo4j database: {database}")

        if args.reset:
            reset_db(driver, database)

        create_schema(driver, database)
        import_kg(driver, database, kg)

        elapsed = time.time() - t0
        print(f"\nImport successfully finished in {elapsed:.2f} seconds.")

        print_graph_summary(driver, database)

    except Exception as exc:
        print(f"\n[ERROR] Failed to load KG into Neo4j: {exc}", file=sys.stderr)
        raise
    finally:
        if driver is not None:
            driver.close()


if __name__ == "__main__":
    main()
