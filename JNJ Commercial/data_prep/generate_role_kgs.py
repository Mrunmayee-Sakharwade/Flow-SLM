#!/usr/bin/env python3
"""
generate_role_kgs.py
Generates the verified Knowledge Graph specification (kg_os.json) and account barriers (account_barriers.json)
for Oncology Sales (OS) on INLEXZO across the 8 real oncology accounts from 'generated_os_training_transcripts 1.xlsx'.
"""

from data_prep.build_os_knowledge_graph import main

if __name__ == "__main__":
    main()
