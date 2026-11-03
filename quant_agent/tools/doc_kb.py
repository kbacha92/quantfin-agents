"""Document stores: two ChromaDB collections with different point-in-time rules.

- "docs"      FACTS (filings, news, notes) from DOCS_DIR. Hard filter on
              every query: pub_date <= as-of. Never exempt.
- "knowledge" TIMELESS METHODOLOGY (textbooks, papers) from KNOWLEDGE_DIR.
              Not date-filtered - citable for methods and definitions only,
              never for facts about the as-of period. Results are marked
              kind="method" with a postdates_as_of flag so the hindsight
              exposure is auditable.

Publication date = YYYY-MM-DD filename prefix, else file mtime (flagged).
Placeholder explainer files (put-*.txt) are never indexed.
"""
import hashlib
import re
from datetime import date
from pathlib import Path

from .. import config
from ..schemas import Evidence
from .base import Tool

_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_EXTS = {".txt", ".pdf", ".html", ".htm"}

# --- Retrieval instrumentation (v9.2) -------------------------------------
# Each retrieval emits a probe record so empty results are self-explaining:
# was the collection populated at the instant of emptiness, how many queries had
# already run in this process, did an immediate retry succeed. Draining these
# (get_probes/clear_probes) into the trace lets the eval bucket empties by
# position-in-run and collection-count-at-emptiness, settling whether got=[] is
# a cold-start race (low count / front-loaded) or inherent ranking variance
# (full count / scattered). The orchestrator drains after each retrieval.
import time as _time

_PROBES: list[dict] = []
_QUERY_COUNTER = {"n": 0}            # process-wide successful+attempted query count
_PROC_START = _time.monotonic()      # for elapsed-since-process-start


def get_probes() -> list[dict]:
    return list(_PROBES)


def clear_probes() -> None:
    _PROBES.clear()


def _record_probe(**fields) -> None:
    _PROBES.append({"elapsed_s": round(_time.monotonic() - _PROC_START, 4), **fields})


def _pub_date(path: Path) -> tuple[date, bool]:
    m = _DATE_RE.search(path.name)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3))), True
    return date.fromtimestamp(path.stat().st_mtime), False


def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    text = path.read_text(errors="ignore")
    if path.suffix.lower() in {".html", ".htm"}:
        text = re.sub(r"<[^>]+>", " ", text)
    return text


def _chunks(text: str, size: int = 1500, overlap: int = 200) -> list[str]:
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + size])
        i += size - overlap
    return [c for c in out if c.strip()]


def _collection(name: str):
    import chromadb

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(name)


def _index_folder(folder: Path, collection_name: str) -> int:
    """Index a corpus folder into the named collection.

    v7.1: content-level idempotency. Each chunk has a deterministic id
    (md5 of collection:file:chunk). Before embedding, we ask the collection which
    of the candidate ids it ALREADY has and skip those - Chroma embeds on every
    upsert, so re-upserting unchanged content would pay the full (slow) embedding
    cost again. Skipping already-present ids makes a second ingest over the same
    corpus near-instant, which is what lets the eval ingest the filings ONCE and
    reuse the index across every case instead of re-embedding per case.
    """
    col = _collection(collection_name)

    # Gather all candidate (id, chunk, metadata) first.
    candidates = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _EXTS:
            continue
        if path.name.startswith("put-"):  # placeholder explainers
            continue
        pub, reliable = _pub_date(path)
        for j, chunk in enumerate(_chunks(_read(path))):
            cid = hashlib.md5(f"{collection_name}:{path.name}:{j}".encode()).hexdigest()
            candidates.append((cid, chunk, {
                "file": path.name, "chunk": j,
                "pub_date": pub.isoformat(),
                "pub_date_int": int(pub.strftime("%Y%m%d")),
                "date_reliable": reliable,
            }))
    if not candidates:
        return 0

    # Which ids are already embedded? Skip them (the expensive part is embedding).
    all_ids = [c[0] for c in candidates]
    try:
        existing = set(col.get(ids=all_ids).get("ids", []))
    except Exception:
        existing = set()

    new = [c for c in candidates if c[0] not in existing]
    if not new:
        return 0  # corpus already fully indexed -> near-instant no-op

    # Upsert only the new chunks (embedding happens here).
    col.upsert(
        ids=[c[0] for c in new],
        documents=[c[1] for c in new],
        metadatas=[c[2] for c in new],
    )
    return len(new)


def ingest() -> int:
    """Index the facts corpus (DOCS_DIR). Idempotent.

    When QUANT_EVAL_SKIP_INGEST is set (the eval prebuilt a shared index and is
    reusing it across cases), this is a no-op: the corpus is already indexed in
    the shared Chroma dir, so re-embedding would defeat the whole optimization.
    """
    import os
    if os.environ.get("QUANT_EVAL_SKIP_INGEST"):
        return 0
    return _index_folder(config.DOCS_DIR, "docs")


def ingest_knowledge() -> int:
    """Index the methodology corpus (KNOWLEDGE_DIR). Idempotent."""
    return _index_folder(config.KNOWLEDGE_DIR, "knowledge")


def _first_row(res: dict) -> tuple[list, list]:
    """Safely unpack a Chroma query result's first (and only) query row.

    Chroma returns {"documents": [[...]], "metadatas": [[...]]}, but on an empty
    collection or empty match it can return [], [[]], or [None]. Indexing [0]
    blindly raises IndexError/TypeError before our friendly 'no matches' message
    fires. This normalizes all of those to a pair of parallel lists (possibly
    empty) so the caller can treat 'no results' as the empty case, not a crash.
    """
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    docs0 = docs[0] if docs and docs[0] else []
    metas0 = metas[0] if metas and metas[0] else []
    return docs0, metas0


def _query_with_retry(collection_name: str, query_texts, n_results, where=None, attempts=3):
    """Query a collection, retrying on an empty result IF the collection is
    actually non-empty.

    Diagnosed failure mode (v5 eval): on ~1 in 8 fresh-subprocess runs, the
    first query against a freshly-ingested collection returns NOTHING even though
    the documents are indexed - an embedding/index cold-start timing hiccup, not
    a relevance miss (when it works, it returns exactly the right chunks). We
    distinguish the two cases by checking collection.count(): if the collection
    has documents but the query returned none, that's the hiccup - re-open the
    collection and retry. If the collection is genuinely empty (or has no docs in
    the date window), we do NOT retry - that's a correct empty result.
    """
    import time

    _QUERY_COUNTER["n"] += 1
    query_index = _QUERY_COUNTER["n"]   # position of this query within the process

    last = None
    attempt_counts = []   # collection.count() observed at each attempt
    empties = 0
    for i in range(attempts):
        col = _collection(collection_name)
        try:
            cnt = col.count()
        except Exception:
            cnt = -1
        res = col.query(query_texts=query_texts, n_results=n_results, where=where)
        docs0, _ = _first_row(res)
        attempt_counts.append(cnt)
        if docs0:
            # success - record whether it took a retry (empties>0 before this)
            _record_probe(collection=collection_name, query_index=query_index,
                          result="hit", attempt=i + 1, attempts_made=i + 1,
                          collection_count=cnt, retried=(i > 0),
                          empties_before_hit=empties,
                          n_results=n_results, has_date_filter=bool(where))
            return res
        empties += 1
        try:
            non_empty = cnt > 0
        except Exception:
            non_empty = False
        if not non_empty:
            # genuinely empty collection (or none in date window) -> correct, no retry
            _record_probe(collection=collection_name, query_index=query_index,
                          result="empty_correct", attempt=i + 1, attempts_made=i + 1,
                          collection_count=cnt, retried=(i > 0),
                          n_results=n_results, has_date_filter=bool(where))
            return res
        last = res
        time.sleep(0.25 * (i + 1))  # brief backoff before re-opening the collection

    # Exhausted all attempts and still empty despite a populated collection:
    # this is the diagnostic case - full collection, query returns nothing.
    _record_probe(collection=collection_name, query_index=query_index,
                  result="empty_despite_populated", attempts_made=attempts,
                  collection_count=attempt_counts[-1] if attempt_counts else -1,
                  collection_counts_per_attempt=attempt_counts, retried=True,
                  n_results=n_results, has_date_filter=bool(where))
    return last if last is not None else res


class DocSearchTool(Tool):
    name = "doc_search"
    description = (
        "doc_search(query: str, k: int = 4) - semantic search over PERIOD FACTS (filings, news, notes). "
        "Only documents published on or before the as-of date are searchable."
    )

    def run(self, as_of, query, k=4, **_):
        res = _query_with_retry(
            "docs",
            query_texts=[str(query)],
            n_results=int(k),
            where={"pub_date_int": {"$lte": int(as_of.strftime("%Y%m%d"))}},
        )
        docs0, metas0 = _first_row(res)
        evidence = []
        for doc, md in zip(docs0, metas0):
            evidence.append(
                Evidence(
                    source_id=f"doc:{md['file']}#chunk{md['chunk']}",
                    store="doc_kb",
                    content=doc,
                    pub_date=date.fromisoformat(md["pub_date"]),
                    pit_safe=bool(md.get("date_reliable", True)),
                    kind="fact",
                )
            )
        if not evidence:
            raise RuntimeError(
                f"No documents matching '{query}' published on or before {as_of}. "
                "Add files to the docs folder and reindex, or fetch filings with edgar_fetch."
            )
        return evidence


class KnowledgeSearchTool(Tool):
    name = "knowledge_search"
    description = (
        "knowledge_search(query: str, k: int = 4) - semantic search over TIMELESS METHODOLOGY "
        "(textbooks, journal articles, formula references). NOT date-filtered. Results support "
        "methods, formulas, and definitions only - never facts about the as-of period; a "
        "post-as-of paper's estimator is usable, its empirical findings are hindsight."
    )

    def run(self, as_of, query, k=4, **_):
        res = _query_with_retry("knowledge", query_texts=[str(query)], n_results=int(k))
        docs0, metas0 = _first_row(res)
        evidence = []
        for doc, md in zip(docs0, metas0):
            pub = date.fromisoformat(md["pub_date"])
            evidence.append(
                Evidence(
                    source_id=f"knowledge:{md['file']}#chunk{md['chunk']}",
                    store="knowledge",
                    content=doc,
                    pub_date=pub,
                    pit_safe=False,
                    kind="method",
                    meta={
                        "postdates_as_of": bool(pub > as_of),
                        "usage_rule": "methods/definitions only - never cite for facts about the as-of period",
                    },
                )
            )
        if not evidence:
            raise RuntimeError(
                f"No knowledge-corpus matches for '{query}'. Add textbooks/papers to the knowledge "
                "folder and reindex (sidebar button)."
            )
        return evidence
