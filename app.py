"""Streamlit chat UI. Put your API keys in keys.py, then launch via run.py."""
import datetime as dt

import streamlit as st

import keys

keys.apply()

from quant_agent import config
from quant_agent.orchestrator import run_query

st.set_page_config(page_title="Quant research agent", layout="wide")
st.title(f"Quant research agent — v{config.APP_VERSION}")
st.caption("Point-in-time, provenance-cited research over prices, filings, and your documents.")

with st.sidebar:
    st.header("Point-in-time")
    as_of = st.date_input("As-of date", value=dt.date.today(), max_value=dt.date.today())
    st.caption("Every retrieval is filtered to data published on or before this date.")
    max_iters = st.slider("Iteration cap", 1, 8, config.MAX_ITERS)
    st.divider()
    st.header("Corpora")
    st.caption(
        f"**Facts** (date-filtered): drop filings/news/notes into `{config.DOCS_DIR}`, "
        "filenames prefixed YYYY-MM-DD. **Methodology** (not date-filtered): drop "
        f"textbooks/papers into `{config.KNOWLEDGE_DIR}` - citable for methods and "
        "definitions only, never period facts."
    )
    if st.button("Reindex docs + knowledge"):
        from quant_agent.tools.doc_kb import ingest, ingest_knowledge
        with st.spinner("Indexing..."):
            n1, n2 = ingest(), ingest_knowledge()
        st.success(f"Indexed {n1} fact chunks, {n2} knowledge chunks.")
    st.caption(f"Generated charts and exports land in `{config.OUTPUTS_DIR}`.")
    st.caption(
        "Tip: you can also just ask the agent to pull filings — e.g. "
        "\"fetch AAPL's latest 10-K and summarize the risk factors\" — "
        "and it will use edgar_fetch + doc_search itself."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if q := st.chat_input("Ask a research question..."):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        status = st.status("Working...", expanded=True)
        ans = run_query(q, as_of, status_cb=status.write, max_iters=max_iters)
        labels = {"ok": "Done", "insufficient_data": "Stopped: insufficient data", "error": "Error"}
        status.update(label=labels[ans.status], state="complete" if ans.status == "ok" else "error", expanded=False)
        st.markdown(ans.text)
        from pathlib import Path as _P
        for art in ans.artifacts:
            if _P(art).exists():
                st.image(art)
        if ans.citations:
            with st.expander(f"Sources ({len(ans.citations)})"):
                for c in ans.citations:
                    st.code(c, language=None)
        st.caption(
            f"status: {ans.status} · iterations: {ans.iterations} · "
            f"trace: data/traces/{ans.trace_id}.jsonl"
        )
    st.session_state.messages.append({"role": "assistant", "content": ans.text})
