# Johnson & Johnson Commercial Oncology: Knowledge Graph System Documentation

---

## 1. Executive Summary

### 1.1 Project Overview
In pharmaceutical commercial operations and medical affairs, cross-functional field teams interact daily with healthcare professionals (HCPs), clinical trial investigators, pharmacists, and institutional account stakeholders. Because pharmaceutical promotion and clinical education are governed by strict FDA, legal, and organizational compliance mandates, each field role operates within clearly defined legal and policy boundaries.

This project delivers a **production-ready Knowledge Graph (KG) deployed on Neo4j Aura Cloud** that structures the **Persona Solid Cancer** operational guidelines (Bladder and Lung cancer commercialization) into an intelligent, queryable, and AI-ready network.

```
┌──────────────────────────────────────────────┐
│                  AT A GLANCE                 │
├──────────────────────────────────────────────┤
│ • Total Graph Nodes:        117              │
│ • Total Graph Edges:        352              │
│ • Roles & Personas:         8 Roles          │
│ • Operational Domains:      7 Domains        │
│ • Governed Rules:           77 Rules         │
│ • Shared Medical Topics:    22 Topics        │
│ • Isolated Nodes:           0 (Zero)         │
│ • Excel Sync Fidelity:      100% (102/102)   │
│ • Compliance Governance:    100% Coverage    │
└──────────────────────────────────────────────┘
```

### 1.2 Core Business Objectives
1. **Transform Static Policy into Active Intelligence**: Transition from flat, unsearchable spreadsheet tables to an interconnected relational network.
2. **Enforce Absolute Compliance Guardrails**: Establish unambiguous boundaries between commercial promotion, non-promotional clinical education, reimbursement access, and scientific medical affairs.
3. **Power Next-Gen AI Assistants (RAG)**: Provide structured, deterministic grounding data for Large Language Models (LLMs) and chatbots to eliminate hallucinations in field interaction summaries.
4. **Instant Cross-Functional Visibility**: Enable commercial leadership and compliance auditors to instantly inspect role permissions, handoff protocols, and topical responsibilities.

---

## 2. End-to-End System Architecture

The data pipeline transforms human-written business rules into a cloud-hosted Neo4j graph through a 4-stage pipeline:

```
┌────────────────────────┐       ┌────────────────────────┐       ┌────────────────────────┐       ┌────────────────────────┐
│      1. EXCEL FILE     │       │      2. JSON SPEC      │       │    3. PYTHON LOADER    │       │     4. NEO4J AURA      │
│  Persona Solid Cancer  │ ────► │  Persona_Solid_Cancer  │ ────► │    neo4j_aura_kg.py    │ ────► │      Cloud Graph       │
│   Roles & Resp.xlsx    │       │         .json          │       │  Batch Cypher Pipeline │       │   119 Nodes / 352 Edges│
└────────────────────────┘       └────────────────────────┘       └────────────────────────┘       └────────────────────────┘
```

### 2.1 File Artifacts in the Solution
* **Source Excel Sheet** (`Persona Solid Cancer_Roles and Responsibility - Bladder and Lung.xlsx`):
  The definitive source document outlining 8 field roles and 102 individual policy and responsibility statements.
* **JSON Schema Document** (`Persona_Solid_Cancer.json`):
  The machine-readable ontological blueprint converting unstructured rows into discrete typed nodes, directed relationships, and metadata attributes.
* **Configuration Environment** (`.env`):
  Secure configuration file managing database credentials (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`) using TLS encryption (`neo4j+ssc://`).
* **High-Performance Python Loader** (`neo4j_aura_kg.py`):
  An idempotent ingestion engine utilizing Cypher batch transactions (`UNWIND`) to create constraints, build indexes, and load all 119 nodes and 352 edges in **~5.5 seconds**.

---

## 3. Data Synchronization: Excel vs. Knowledge Graph

To ensure complete regulatory integrity, every item in the Excel document was audited against the Knowledge Graph.

### 3.1 Verification Matrix (100% Verbatim Match)

| Role Code | Role Name | Excel Numbered Items | Graph Rules & Variants | Excel Match Fidelity |
| :--- | :--- | :---: | :---: | :---: |
| **`OS`** | Oncology Sales Representative | 15 | 15 | **100% Exact Match** |
| **`OCE`** | Oncology Clinical Educator | 11 | 11 | **100% Exact Match** |
| **`FRM`** | Field Reimbursement Manager | 19 | 19 | **100% Exact Match** |
| **`MSL`** | Medical Science Liaison | 9 | 9 | **100% Exact Match** |
| **`KAM`** | Key Account Manager | 12 | 12 | **100% Exact Match** |
| **`UBAM`** | Urology Business Account Manager | 11 | 11 | **100% Exact Match** |
| **`PMAM`** | Precision Medicine Account Manager | 13 | 13 | **100% Exact Match** |
| **`TLL`** | Thought Leader Liaison | 12 | 12 | **100% Exact Match** |
| **TOTAL** | **8 Personas** | **102 Items** | **102 Items** | **100% (0 Mismatches)** |

### 3.2 How Excel Rows Map to Graph Entities
* **8 Role Descriptions**: Mapped directly to `Role.description` properties.
* **44 Core Responsibilities**: Mapped to `CoreResponsibility` nodes connected via `HAS_RESPONSIBILITY`.
* **13 Exclusion Rules**: Mapped to `ExclusionRule` nodes connected via `HAS_EXCLUSION_RULE`.
* **10 Documentation Rules**: Mapped to `DocumentationRule` nodes connected via `HAS_DOCUMENTATION_RULE`.
* **10 Collaboration Rules**: Mapped to `CollaborationRule` nodes connected via `HAS_COLLABORATION_RULE`.
* **25 Policy Variations**: Mapped directly to relationship metadata (`variant_text`) on `GOVERNED_BY` and `COLLABORATES_WITH` edges.

---

## 4. Persona Profiles & Operational Boundaries

```
                                  ORGANIZATIONAL ECOSYSTEM
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                                   GLOBAL COMPLIANCE                                    │
 │                    PHI/PII Protection • MIR Capture • Toxicity Control                  │
 └─────────────┬─────────────────────────────┬────────────────────────────┬───────────────┘
               │ (Governs)                   │ (Governs)                  │ (Governs)
               ▼                             ▼                            ▼
 ┌───────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐
 │     COMMERCIAL SALES      │ │      MEDICAL AFFAIRS      │ │      PATIENT ACCESS       │
 │ • OS (Oncology Specialist)│ │ • MSL (Medical Liaison)   │ │ • FRM (Reimbursement)     │
 │ • KAM (Key Account Mgr)   │ │ • OCE (Clinical Educator) │ │   Prior Auth, Copay &      │
 │ • UBAM (Urology Acct Mgr) │ │   Scientific exchange,    │ │   Coverage Assistance      │
 │ • PMAM (Precision Med)    │ │   nurse training, trial   │ │                           │
 │ • TLL (Thought Leader)    │ │   evidence, education     │ │                           │
 └───────────────────────────┘ └───────────────────────────┘ └───────────────────────────┘
```

### 4.1 Detailed Persona Breakdown

#### 1. Oncology Sales Representative (`OS`)
* **Primary Domain**: Commercial / promotional engagement
* **Mandate**: Customer-facing commercial engagement with oncologists, hematologists, and pharmacists to present approved product positioning, indications, clinical trial efficacy, and dosing within on-label promotional guidelines.
* **In-Scope Topics**: Dosing administration (on-label), treatment sequencing, efficacy & safety product info, competitive landscape, relationship building, territory engagement planning, stakeholder management, cross-functional collaboration.
* **Strict Exclusions**: Adverse event intake/management (must triage to Medical/Safety), clinical toxicity management, off-label treatment sequencing.

#### 2. Oncology Clinical Educator (`OCE`)
* **Primary Domain**: Non-promotional clinical education
* **Mandate**: Non-promotional clinical education for clinic nurses, infusion staff, and pharmacists regarding product administration, dose preparation, safe handling, and clinical side-effect management.
* **In-Scope Topics**: Disease state education, dosing administration, toxicity management, adverse events & safety, product info, treatment sequencing.
* **Strict Exclusions**: Commercial pricing negotiations, reimbursement patient assistance programs, promotional sales messaging.

#### 3. Field Reimbursement Manager (`FRM`)
* **Primary Domain**: Patient access & reimbursement
* **Mandate**: Assisting healthcare provider offices and billing specialists with navigating complex payer coverage policies, prior authorizations (PAs), appeals, co-pay assistance, and approved reimbursement support programs.
* **In-Scope Topics**: Prior authorization, payer coverage, reimbursement, affordability patient support, patient access, stakeholder management.
* **Strict Exclusions (Highest Boundary)**: Clinical efficacy discussions, medical trial evidence, biomarker testing recommendations, dosing changes, clinical toxicity management.

#### 4. Medical Science Liaison (`MSL`)
* **Primary Domain**: Medical / scientific affairs
* **Mandate**: Peer-to-peer non-promotional scientific exchange with Key Opinion Leaders (KOLs) and oncologists. Authorized to discuss complex trial evidence, off-label clinical research, biomarker data, and unsolicited scientific questions.
* **In-Scope Topics**: Biomarker testing, clinical trial evidence, disease state education, dosing administration, adverse events safety, treatment sequencing, KOL engagement.
* **Strict Exclusions**: 0 exclusions (broadest clinical latitude to answer unsolicited scientific questions; non-promotional).

#### 5. Key Account Manager (`KAM`) & Urology Business Account Manager (`UBAM`)
* **Primary Domain**: Institutional account strategy
* **Mandate**: Developing strategic business plans for hospital networks, large health systems, and urology clinics to address institutional contracting, customer business needs, and organizational adoption.
* **In-Scope Topics**: Account planning, relationship building, stakeholder management, cross-functional collaboration, competitive landscape.
* **Strict Exclusions**: Clinical toxicity management, direct adverse event reporting, patient-specific medical advice.

#### 6. Precision Medicine Account Manager (`PMAM`)
* **Primary Domain**: Precision medicine & diagnostics
* **Mandate**: Collaborating with pathology laboratories, hospital molecular teams, and diagnostic specialists to optimize biomarker testing protocols and patient testing journeys.
* **In-Scope Topics**: Biomarker testing, diagnostic workflows, account planning, stakeholder management.
* **Strict Exclusions**: Treatment recommendations, drug efficacy discussions, clinical dosing advice.

#### 7. Thought Leader Liaison (`TLL`)
* **Primary Domain**: KOL / thought leader engagement
* **Mandate**: Managing non-promotional relationships with top-tier national medical experts, advisory board planning, speaker bureau execution, and congress engagements.
* **In-Scope Topics**: KOL engagement, speaker bureau management, clinical trial evidence, territory planning.
* **Strict Exclusions**: Direct promotional sales negotiation, patient clinical management.

---

## 5. Knowledge Graph Schema & Ontology

The graph adheres to strict ontological rules in Neo4j.

### 5.1 Node Labels & Schema Taxonomy

```
                          ┌──────────────────────────┐
                          │         :KGNode          │ (Base label on all 117 nodes)
                          └─────────────┬────────────┘
                                        │
        ┌──────────────┬──────────────┬─┴────────────┬──────────────┬──────────────┐
        ▼              ▼              ▼              ▼              ▼              ▼
     [:Role]       [:Domain]       [:Topic]    [:ComplianceRule] [:CoreResp]  [:ExclusionRule]
     (8 nodes)     (7 nodes)      (22 nodes)       (3 nodes)     (44 nodes)      (13 nodes)
```

| Label | Count | Primary Properties | Business Purpose |
| :--- | :---: | :--- | :--- |
| **`Role`** | 8 | `id`, `name`, `full_name`, `description`, `primary_domain` | Field persona definitions |
| **`Domain`** | 7 | `id`, `name` | Organizational functional areas |
| **`Topic`** | 22 | `id`, `name` | Core medical and business conversation subjects |
| **`ComplianceRule`** | 3 | `id`, `rule_type`, `severity`, `action`, `summary` | Universal regulatory mandates |
| **`CoreResponsibility`**| 44 | `id`, `text`, `domain`, `topics` | Positive role duties ("What I must do") |
| **`ExclusionRule`** | 13 | `id`, `text`, `domain`, `topics` | Hard role boundaries ("What is forbidden") |
| **`DocumentationRule`**| 10 | `id`, `text`, `domain`, `topics` | Interaction logging protocols |
| **`CollaborationRule`**| 10 | `id`, `text`, `domain`, `topics` | Cross-functional engagement rules |

### 5.2 Relationship Types (352 Edges)

| Relationship Type | Count | Source Label $\rightarrow$ Target Label | Meaning |
| :--- | :---: | :--- | :--- |
| **`OPERATES_IN`** | 8 | `Role` $\rightarrow$ `Domain` | Maps each role to its operational department |
| **`GOVERNED_BY`** | 24 | `Role` $\rightarrow$ `ComplianceRule` | Universal compliance governance (8 roles $\times$ 3 rules) |
| **`HAS_RESPONSIBILITY`**| 44 | `Role` $\rightarrow$ `CoreResponsibility` | Connects roles to their positive responsibilities |
| **`HAS_EXCLUSION_RULE`**| 13 | `Role` $\rightarrow$ `ExclusionRule` | Connects roles to their strict exclusions |
| **`HAS_DOCUMENTATION_RULE`**| 10 | `Role` $\rightarrow$ `DocumentationRule` | Connects roles to documentation mandates |
| **`HAS_COLLABORATION_RULE`**| 10 | `Role` $\rightarrow$ `CollaborationRule` | Connects roles to collaboration mandates |
| **`HAS_IN_SCOPE_TOPIC`**| 53 | `Role` $\rightarrow$ `Topic` | Direct topical permissions |
| **`HAS_OUT_OF_SCOPE_TOPIC`**| 29 | `Role` $\rightarrow$ `Topic` | Direct topical exclusions |
| **`ABOUT_TOPIC`** | 70 | `CoreResponsibility` $\rightarrow$ `Topic` | Maps duties to medical/business subjects |
| **`EXCLUDES_TOPIC`** | 29 | `ExclusionRule` $\rightarrow$ `Topic` | Maps negative boundaries to subjects |
| **`DOCUMENTS_TOPIC`**| 19 | `DocumentationRule` $\rightarrow$ `Topic` | Maps documentation requirements to subjects |
| **`RELATES_TO_TOPIC`**| 22 | `CollaborationRule` $\rightarrow$ `Topic` | Maps collaboration triggers to subjects |
| **`COLLABORATES_WITH`**| 21 | `Role` $\rightarrow$ `Role` | Cross-functional referral and handoff paths |

---

## 6. Regulatory & Compliance Framework

The graph enforces 3 mandatory compliance policies across all 8 personas:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        UNIVERSAL PHARMACEUTICAL GUARDRAILS                             │
├──────────────────────────┬───────────────────────────┬─────────────────────────────────┤
│ 1. PHI / PII Protection  │ 2. MIR Handling Protocol  │ 3. Toxicity Language Control    │
├──────────────────────────┼───────────────────────────┼─────────────────────────────────┤
│ • Rule: privacy_phi_pii  │ • Rule: mir_handling      │ • Rule: toxicity_control        │
│ • Severity: Mandatory    │ • Severity: Mandatory     │ • Severity: Mandatory           │
│ • Policy: Total exclusion│ • Policy: Unsolicited     │ • Policy: Strict replacement of │
│   and redaction of all   │   clinical questions must │   the word "toxicity" with      │
│   patient-identifying    │   be recorded as admin-   │   "safety concerns" in all      │
│   information from notes │   only logs and triaged.  │   commercial interaction logs.  │
└──────────────────────────┴───────────────────────────┴─────────────────────────────────┘
```

---

## 7. Graph Topology & Connectivity Analysis

### 7.1 Complete Topological Health Check
* **Zero Isolated Nodes (0 Isolated)**:
  * 100% of nodes in the graph are actively connected with at least one incoming or outgoing edge.
  * Orphaned/unreferenced topic dictionary entries have been removed so the graph strictly reflects the Excel personas and duties.
* **Leaf Duty Nodes (15 Rules)**:
  * 11 `CoreResponsibility`, 2 `ExclusionRule`, and 2 `DocumentationRule` nodes connect back to their owning Role but have no outgoing arrow to a specific Topic.
  * **Why**: These represent general operational and administrative policies (e.g., *"Documenting customer interactions compliantly"*) rather than disease-specific subjects.
* **Dual-Scope Topics (6 Topic Overlaps)**:
  * Topics like `dosing administration` for `OS` appear both in-scope and out-of-scope.
  * **Why**: Reflects conditional permissions (On-label dosing = In-Scope; Off-label adjustments = Out-of-Scope).

---

## 8. Cypher Query Playbook for Stakeholders

Run these queries in **Neo4j Browser** to extract immediate business insights:

### Query 1: Total Graph Counts and Label Breakdown
```cypher
MATCH (n:KGNode)
RETURN n.kg_label AS Entity_Type, count(n) AS Total_Count
ORDER BY Total_Count DESC
```

### Query 2: Single Role 360-Degree Star View (e.g., Sales Rep `OS`)
```cypher
MATCH (r:Role {name: 'OS'})-[rel]->(target)
RETURN r, rel, target
```

### Query 3: Who is Permitted to Discuss a Given Topic? (e.g., Prior Authorization)
```cypher
MATCH (r:Role)-[:HAS_IN_SCOPE_TOPIC]->(t:Topic {name: 'prior authorization'})
RETURN r.name AS Role_Code, r.full_name AS Role_Title, t.name AS Allowed_Topic
```

### Query 4: Cross-Functional Handoff Map (Who collaborates with whom?)
```cypher
MATCH (from:Role)-[r:COLLABORATES_WITH]->(to:Role)
RETURN from.name AS Initiating_Role, to.name AS Receiving_Role, r.properties_json AS Protocol
ORDER BY Initiating_Role
```

### Query 5: Full Compliance Policy Audit
```cypher
MATCH (r:Role)-[:GOVERNED_BY]->(c:ComplianceRule)
RETURN r.name AS Role, c.name AS Policy_Name, c.severity AS Severity, c.summary AS Rule_Summary
ORDER BY r.name
```

---

## 9. Strategic Value & Future Roadmap

```
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                             AI & BUSINESS VALUE MATRIX                                 │
 ├──────────────────────────┬─────────────────────────────┬───────────────────────────────┤
 │ Capability               │ Traditional Spreadsheet     │ Neo4j Knowledge Graph         │
 ├──────────────────────────┼─────────────────────────────┼───────────────────────────────┤
 │ Searchability            │ Keyword search only         │ Multi-hop semantic traversal  │
 │ Compliance Auditing      │ Manual spot checks          │ Automated sub-second queries  │
 │ AI Grounding (RAG)       │ Hallucination-prone text    │ Deterministic graph context   │
 │ Scalability              │ Static flat tables          │ Dynamic, extensible ontology  │
 └──────────────────────────┴─────────────────────────────┴───────────────────────────────┘
```

### Strategic Recommendations:
1. **Integrate into CRM / Call-Planning AI**:
   Connect the Knowledge Graph to field reporting tools (e.g., Veeva CRM) to automatically validate field notes before submission.
2. **Expand Therapeutic Areas**:
   Replicate this structure across additional oncology portfolios (e.g., Hematology, Prostate, Lung) using the same high-speed Python ingestion pipeline.
3. **Automated Handoff Routing**:
   Build an automated triage bot that detects off-label inquiries in sales logs and creates instant referral tickets for the designated MSL.

---

## 10. Frequently Asked Questions (FAQ)

**Q: Is any information from the original Excel sheet missing?**  
**A:** No. Every single role description and all 102 numbered duty lines from the Excel file were verified with a 100% exact match.

**Q: Can this graph be modified if business rules change?**  
**A:** Yes. Updating the source JSON or Excel and re-running `python neo4j_aura_kg.py` will update the database cleanly and idempotently in ~5 seconds.

**Q: How does this help our AI and LLM initiatives?**  
**A:** LLMs often hallucinate or confuse permissions across roles. By feeding the LLM exact paths from this Knowledge Graph (GraphRAG), the AI is guaranteed to adhere to J&J's compliance policies.
