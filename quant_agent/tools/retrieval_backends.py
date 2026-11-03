"""Pluggable retrieval backends.

The agent retrieves PERIOD FACTS through a backend. 

  - RagBackend   : semantic similarity over the vector store (nondeterministic;
                   the same query can surface different chunks run-to-run).
  - SqlBackend   : exact queries over a relational table of facts (deterministic;
                   the same query always returns the same rows).
  - GraphBackend : traversal over an in-memory fact graph (deterministic; same
                   start + relation => same neighbours every time).

All three:
  * return the same Evidence shape, so they are drop-in interchangeable;
  * apply the SAME point-in-time filter (pub_date <= as_of), so the guardrail is
    identical regardless of backend;
  * are REAL working skeletons over the ACME facts - minimal, not mocked.

"""
from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

from ..schemas import Evidence


class RetrievalBackend(ABC):
    """Common interface. A backend turns (query, as_of) into PIT-filtered Evidence."""
    name: str = ""

    @abstractmethod
    def retrieve(self, query: str, as_of: date, k: int = 4) -> list[Evidence]: ...

    @staticmethod
    def _pit_ok(pub: date, as_of: date) -> bool:
        return pub <= as_of


# --------------------------------------------------------------------------
# RAG backend - delegates to the existing vector store (nondeterministic).
# --------------------------------------------------------------------------
class RagBackend(RetrievalBackend):
    name = "rag"

    def retrieve(self, query: str, as_of: date, k: int = 4) -> list[Evidence]:
        # Reuse the existing doc_kb semantic search verbatim, so "rag" is exactly
        # today's behaviour - the nondeterministic baseline.
        from .doc_kb import _query_with_retry, _first_row
        res = _query_with_retry(
            "docs", query_texts=[str(query)], n_results=int(k),
            where={"pub_date_int": {"$lte": int(as_of.strftime("%Y%m%d"))}},
        )
        docs0, metas0 = _first_row(res)
        out = []
        for doc, md in zip(docs0, metas0):
            out.append(Evidence(
                source_id=f"doc:{md['file']}#chunk{md['chunk']}",
                store="doc_kb", content=doc,
                pub_date=date.fromisoformat(md["pub_date"]),
                pit_safe=bool(md.get("date_reliable", True)), kind="fact",
            ))
        return out


# --------------------------------------------------------------------------
# SQL backend - exact queries over a relational facts table (deterministic).
# --------------------------------------------------------------------------
class SqlBackend(RetrievalBackend):
    name = "sql"

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def retrieve(self, query: str, as_of: date, k: int = 4) -> list[Evidence]:
        if not self.db_path.exists():
            raise RuntimeError(f"SQL facts DB not found: {self.db_path}. Run build_fact_stores.py.")
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        # Deterministic retrieval: match query terms against the indexed
        # subject/predicate/text columns, PIT-filter by pub_date, stable ordering.
        terms = [t.lower() for t in str(query).split() if len(t) > 2]
        like = " OR ".join(["lower(subject) LIKE ? OR lower(predicate) LIKE ? OR lower(text) LIKE ?"
                            for _ in terms]) or "1=1"
        params = []
        for t in terms:
            params += [f"%{t}%", f"%{t}%", f"%{t}%"]
        sql = (
            "SELECT * FROM facts WHERE pub_date <= ? AND (" + like + ") "
            "ORDER BY pub_date DESC, id ASC LIMIT ?"
        )
        rows = con.execute(sql, [as_of.isoformat(), *params, int(k)]).fetchall()
        con.close()
        out = []
        for r in rows:
            out.append(Evidence(
                source_id=f"sql:{r['source_file']}#row{r['id']}",
                store="sql", content=r["text"],
                pub_date=date.fromisoformat(r["pub_date"]),
                pit_safe=True, kind="fact",
            ))
        return out


# --------------------------------------------------------------------------
# Graph backend - traversal over an in-memory fact graph (deterministic).
# --------------------------------------------------------------------------
class GraphBackend(RetrievalBackend):
    name = "graph"

    def __init__(self, graph_path: Path):
        self.graph_path = Path(graph_path)
        self._graph = None

    def _load(self):
        if self._graph is None:
            import json
            if not self.graph_path.exists():
                raise RuntimeError(f"graph not found: {self.graph_path}. Run build_fact_stores.py.")
            self._graph = json.loads(self.graph_path.read_text())
        return self._graph

    def retrieve(self, query: str, as_of: date, k: int = 4) -> list[Evidence]:
        g = self._load()
        nodes = g["nodes"]          # id -> {label, ...}
        edges = g["edges"]          # list of {s, p, o, text, pub_date, source_file}
        terms = [t.lower() for t in str(query).split() if len(t) > 2]

        # Deterministic traversal: find seed nodes whose label matches a query
        # term, then collect edges incident to those nodes (1-hop), PIT-filtered.
        seeds = [nid for nid, n in nodes.items()
                 if any(t in n["label"].lower() for t in terms)]
        seedset = set(seeds)
        hits = []
        for e in edges:
            if e["s"] in seedset or e["o"] in seedset:
                if date.fromisoformat(e["pub_date"]) <= as_of:
                    hits.append(e)
        # stable order: by pub_date then a deterministic key
        hits.sort(key=lambda e: (e["pub_date"], e["s"], e["p"], e["o"]))
        out = []
        for i, e in enumerate(hits[:k]):
            out.append(Evidence(
                source_id=f"graph:{e['source_file']}#{e['s']}-{e['p']}-{e['o']}",
                store="graph", content=e["text"],
                pub_date=date.fromisoformat(e["pub_date"]),
                pit_safe=True, kind="fact",
            ))
        return out


def get_backend(name: str):
    """Factory. Reads paths from config so the eval can select a backend by name."""
    from .. import config
    name = (name or "rag").lower()
    if name == "rag":
        return RagBackend()
    if name == "sql":
        return SqlBackend(config.DATA_DIR / "facts.db")
    if name == "graph":
        return GraphBackend(config.DATA_DIR / "facts_graph.json")
    raise ValueError(f"unknown backend '{name}' (use rag|sql|graph)")
