#!/usr/bin/env python3
"""Build the SQL and Graph fact stores from a shared seed of ACME facts (v8.1).

Both stores are populated from the SAME structured fact list, so the SQL, Graph,
and RAG backends all retrieve the same underlying facts - the comparison is about
the RETRIEVAL MECHANISM (deterministic exact/traversal vs nondeterministic
semantic), not about different data.

    python eval/build_fact_stores.py

Outputs:
    data/facts.db          (SQLite: one row per fact, PIT-filterable by pub_date)
    data/facts_graph.json  (nodes + edges; same facts as triples)

SCOPE (honesty): the seed facts below are a curated, demonstration-scale subset
mirroring the ACME corpus (the 10-K, 8-K, analyst note, press release). A
production version would EXTRACT these from the filing prose (NLP relation
extraction) - deliberately out of scope for the scaffold. The point is three
genuinely-working backends over a shared fact set.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Shared seed: (subject, predicate, object/value, free-text, pub_date, source_file)
# pub_date drives the point-in-time filter, identical to the RAG store.
FACTS = [
    ("ACME", "fiscal_2024_revenue", "4.20 billion",
     "ACME reported audited fiscal 2024 revenue of 4.20 billion dollars.",
     "2024-11-01", "2024-11-01_ACME_10-K.txt"),
    ("ACME", "fiscal_2024_net_income", "512 million",
     "ACME reported fiscal 2024 net income of 512 million dollars.",
     "2024-11-01", "2024-11-01_ACME_10-K.txt"),
    ("ACME", "supplier_concentration", "single supplier Region B",
     "ACME sourced 90 percent of its specialty resin from a single supplier in Region B.",
     "2024-11-01", "2024-11-01_ACME_10-K.txt"),
    ("ACME", "customer_concentration", "38 percent",
     "ACME's largest customer accounted for 38 percent of revenue.",
     "2024-11-01", "2024-11-01_ACME_10-K.txt"),
    ("ACME", "q4_2025_channel_softness", "Region B",
     "ACME's Q4 fiscal 2025 8-K noted channel softness in Region B.",
     "2025-02-10", "2025-02-10_ACME_8-K.txt"),
    ("ACME", "guidance_2025", "withdrew",
     "In its 8-K ACME withdrew prior full-year guidance citing uncertainty.",
     "2025-02-10", "2025-02-10_ACME_8-K.txt"),
    ("ACME", "analyst_revenue_estimate", "4.50 billion",
     "A third-party analyst note estimated ACME fiscal 2024 revenue at approximately 4.50 billion, a derived estimate not an audited number.",
     "2025-01-15", "2025-01-15_ACME_analyst_note.txt"),
    ("ACME", "acquired", "Beacon Materials",
     "ACME announced the acquisition of Beacon Materials for 800 million dollars in cash.",
     "2025-05-20", "2025-05-20_ACME_press.txt"),
    ("Beacon Materials", "provides", "specialty resin supply",
     "Beacon Materials gives ACME a second specialty-resin supplier, reducing single-supplier concentration risk.",
     "2025-05-20", "2025-05-20_ACME_press.txt"),
]


def build_sql():
    DATA.mkdir(parents=True, exist_ok=True)
    db = DATA / "facts.db"
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject TEXT, predicate TEXT, object TEXT, text TEXT,
        pub_date TEXT, source_file TEXT
    )""")
    con.executemany(
        "INSERT INTO facts (subject,predicate,object,text,pub_date,source_file) VALUES (?,?,?,?,?,?)",
        FACTS,
    )
    con.execute("CREATE INDEX idx_pub ON facts(pub_date)")
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    con.close()
    return n


def build_graph():
    DATA.mkdir(parents=True, exist_ok=True)
    nodes = {}
    edges = []

    def node(label):
        nid = label.lower().replace(" ", "_")
        nodes.setdefault(nid, {"label": label})
        return nid

    for s, p, o, text, pub, src in FACTS:
        sid = node(s)
        oid = node(str(o))
        edges.append({"s": sid, "p": p, "o": oid, "text": text,
                      "pub_date": pub, "source_file": src})
    graph = {"nodes": nodes, "edges": edges}
    out = DATA / "facts_graph.json"
    out.write_text(json.dumps(graph, indent=2))
    return len(nodes), len(edges)


def main():
    n_sql = build_sql()
    n_nodes, n_edges = build_graph()
    print(f"SQL facts.db: {n_sql} rows")
    print(f"Graph facts_graph.json: {n_nodes} nodes, {n_edges} edges")
    print("Both populated from the same shared fact seed.")


if __name__ == "__main__":
    main()
