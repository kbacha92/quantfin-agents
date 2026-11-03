# Golden evaluation set — quant research agent

A 50-case golden set for regression-testing the agent's **behavior**, not just
its arithmetic. The hard part of evaluating this system is whether the
point-in-time and provenance guardrails hold, so most cases assert on *what the
agent is and isn't allowed to do* given a known corpus state.

## What's here

    eval/
      build_snapshots.py        generates frozen price CSVs + ground-truth quant values
      build_golden.py           assembles golden.jsonl from the corpus + quant truth
      validate_golden.py        offline self-consistency + exact-value check (no API key)
      golden_dataset/
        golden.jsonl            the 50 cases (one JSON object per line)
        manifest.json           counts, required snapshots, schema fields
        expected_quant.json     ground-truth quant values (source of pinned numbers)
        snapshots/              frozen yfinance CSVs, replayed offline by load_prices
        seed_docs/              FACTS corpus (dated filenames drive pub_date)
        seed_knowledge/         METHODOLOGY corpus (timeless; one post-as-of paper)

## Coverage (50 cases)

| Area         | Cases | What it checks |
|--------------|-------|----------------|
| pit          | 15    | as-of filtering; post-as-of docs invisible; boundary dates inclusive/exclusive; no future leakage in summaries; empty-corpus = no hallucination |
| provenance   | 11    | every period fact carries a citation; primary filings outrank derived estimates on conflict; conflicts surfaced not hidden; no fabricated citations |
| quant        | 15    | exact indicator/drawdown/correlation values via frozen snapshots; error paths (bad indicator, single-ticker correlation); numbers reported not recomputed; chart artifacts |
| methodology  | 9     | definitions answered from knowledge corpus; METHOD sources never cited as period facts; post-as-of paper's method usable but findings flagged as hindsight |

## Why it runs offline

`load_prices()` replays `data/api_cache/yfinance_<TICKER>_<asof>_<lookback>d.csv`
for any **past** as-of date instead of calling yfinance. The snapshots in
`snapshots/` use exactly those filenames, so the whole price/indicator/chart
path is deterministic with no network. The pinned quant values in each case were
computed by calling the *same* indicator functions the agent uses
(`build_snapshots.py`), then re-verified through the real `QuantIndicatorTool`
(`validate_golden.py`) — so the golden numbers cannot drift from the
implementation.

The as-of date for all quant cases is **2025-03-14** (a Friday, in the past so
snapshots are treated as immutable).

## Record schema

Each line of `golden.jsonl` is one case:

| field | meaning |
|-------|---------|
| `id` | unique case id (e.g. `pit_02_future_8k_invisible`) |
| `area` | `pit` \| `provenance` \| `quant` \| `methodology` |
| `query` | the user question |
| `as_of` | ISO date for the point-in-time vantage |
| `seed_docs` | FACTS-corpus filenames that must be indexed before the run |
| `seed_knowledge` | METHODOLOGY-corpus filenames to index |
| `snapshots` | price snapshot CSVs to stage into `api_cache` before the run |
| `expected_status` | `ok` or `insufficient_data` |
| `expect_tools_any` / `expect_tools_all` | tool names that should fire |
| `expect_citation_source_ids` | exact source_ids that MUST appear in citations |
| `expect_citation_store_any` | ≥1 citation from at least one of these stores |
| `forbid_citation_store` | NO citation may come from these stores |
| `must_contain` / `must_not_contain` | case-insensitive substring asserts on answer text |
| `expected_value` / `value_tolerance` | numeric ground truth (quant cases) |
| `expect_artifact` | answer must reference a saved chart file |
| `rationale` | the guardrail/behavior this case exists to catch |

## Regenerate the dataset

    PYTHONPATH=. python eval/build_snapshots.py    # snapshots + expected_quant.json
    PYTHONPATH=. python eval/build_golden.py       # golden.jsonl + manifest.json

## Validate (offline, no API key)

    PYTHONPATH=. python eval/validate_golden.py

Confirms every referenced file exists, the schema is sane, and every pinned
quant value matches the real tool's output exactly. Exit 0 = consistent.

## Running the agent against it (the harness — needs an API key)

`run_harness.py` runs each case against the live agent and scores it. It calls
the LLM, so it needs a working `ANTHROPIC_API_KEY` (launch the project once via
`run.py`, or `export ANTHROPIC_API_KEY=...`). Token cost scales with cases ×
iterations.

    PYTHONPATH=. python eval/run_harness.py                 # all 50 cases
    PYTHONPATH=. python eval/run_harness.py --area pit      # one area
    PYTHONPATH=. python eval/run_harness.py --id pit_02_future_8k_invisible
    PYTHONPATH=. python eval/run_harness.py --limit 5 --verbose
    PYTHONPATH=. python eval/run_harness.py --report eval/last_run.json

Exit code is 0 iff every selected case passes.

### Head-to-head vs a vanilla model (--baseline + compare_arms.py)

To measure what the agent scaffolding actually adds, run the SAME 50 cases
through a vanilla model - one LLM call, no tools, no corpus, no point-in-time
enforcement - scored by the SAME scorer:

    # agent arm (your normal baseline)
    PYTHONPATH=. python eval/run_harness.py --repeat 3 --report eval/agent.json
    # vanilla arm
    PYTHONPATH=. python eval/run_harness.py --baseline --repeat 3 --report eval/vanilla.json
    # diff them into a per-category head-to-head
    PYTHONPATH=. python eval/compare_arms.py eval/agent.json eval/vanilla.json

The comparison reports, per area, how many cases each arm passed, which cases
the agent won (passed where vanilla failed), and which - if any - the vanilla
model won. Read it per-category, never as a single aggregate: the expected and
honest pattern is that the vanilla model matches on definitional/methodology
questions (no tools needed) and collapses on point-in-time, grounded
computation, and provenance (where retrieval, the as-of guardrail, exact tool
math, and real source_ids are doing the work). A vanilla "win" almost always
means agent over-refusal or an over-strict assertion, not a real model
advantage - worth inspecting either way.

Both arms use the same model (config.MODEL), so the comparison isolates the
scaffolding, not the model. Fairness notes: the vanilla arm is TOLD the as-of
date and asked to answer honestly and cite sources if it has them - it simply
has no tools or corpus to do so, which is the point. For pure-computation
questions a vanilla model is at a structural disadvantage (it never sees the
price data); that gap is real and worth showing, but label it as a
retrieval/tool gap, not a reasoning gap.

### Diagnosing retrieval nondeterminism (diagnose_retrieval.py)

Several multi-document cases are FLAKY because semantic search sometimes
surfaces the chunk the answer needs and sometimes doesn't - the agent reasons
correctly when it gets the chunk, and honestly reports insufficient when it
misses. Before changing retrieval (e.g. raising `k`), diagnose *why* a case
misses:

    PYTHONPATH=. python eval/diagnose_retrieval.py prov_02_primary_outranks_derived 5

For each run it prints the final status and which document chunks were cited,
then summarizes. The reading guide it prints distinguishes three causes, each
with a different fix:
- retrieved NOTHING -> the query missed entirely (recall problem)
- cited the wrong chunk -> ranking/chunking problem
- cited the right chunk but still insufficient -> evaluator, not retrieval

Look first; change retrieval second.

### Nondeterminism: measure stability, not a single sample (--repeat)

The agent calls an LLM, so a borderline case can pass on one run and fail on the
next - the same case run five times might give three `insufficient_data` and two
`ok`. A single 50-case run is therefore ONE SAMPLE of a distribution, not a fixed
score; do not read a one-off "38/50" as the agent's true pass rate.

Use `--repeat N` to run each case N times and report a pass-rate per case. A case
is classified PASS (passed all N), FAIL (failed all N), or FLAKY (mixed - the
important signal):

    PYTHONPATH=. python eval/run_harness.py --repeat 5 --report eval/stability.json
    PYTHONPATH=. python eval/run_harness.py --id pit_02_future_8k_invisible --repeat 5

FLAKY cases are where the agent's behavior is unreliable, which is usually more
actionable than a clean FAIL. The report's `flaky` list and per-case
`passes/runs` capture this. When comparing an agent change against a baseline,
compare `--repeat`'d pass-rates, not single runs, or measurement noise will
swamp the effect you are trying to detect.

### How it works

Each case runs in its **own subprocess with its own temp workspace**, so there
is zero cross-case leakage:

1. A fresh temp dir is created; `QUANT_DATA_DIR` and `QUANT_KNOWLEDGE_DIR` are
   set to point at it *before* `quant_agent.config` is imported (config binds
   its paths from the env at import time). `_run_one_case.py` is the subprocess
   entry point.
2. The case's `seed_docs` are copied into `<data>/docs/`, `seed_knowledge` into
   `<knowledge>/`, then `ingest()` / `ingest_knowledge()` index them into an
   isolated Chroma store.
3. The case's `snapshots` are copied into `<data>/api_cache/` so `load_prices`
   replays them offline (no yfinance call).
4. `run_query(query, as_of)` runs the agent.
5. Which tools actually fired is recovered from the run's trace JSONL
   (`tool_ok` records), since `Answer` doesn't carry that directly.
6. `score()` checks the returned `Answer` against every assertion field:
   status, `expect_tools_all` / `expect_tools_any`, exact
   `expect_citation_source_ids`, `forbid_citation_store` /
   `expect_citation_store_any` (mapping each cited `source_id` prefix to its
   store), `must_contain` / `must_not_contain`, `expect_artifact`, and
   `expected_value` within `value_tolerance`.

Because steps 1–3 fully determine the corpus and price data, every case is
reproducible run-to-run except for LLM phrasing nondeterminism — which the
substring and citation asserts are written to tolerate.

### Offline-strict mode (v4)

The harness sets `QUANT_OFFLINE_ONLY=1` for every case, so `load_prices` never
touches the network: it replays the staged snapshot or raises. This guarantees
the eval cannot silently measure live data even if the agent mutates a ticker
to one without a snapshot (the SYN->SYNB bug v3 surfaced). If you add a quant
case for a new ticker, generate its snapshot first (`build_snapshots.py`) or the
case will fail closed with an "offline-strict mode: no snapshot" error - which
is the intended safety behavior, not a harness bug.

### First-run note (Chroma model download)

The first `ingest()` on a machine triggers a one-time download of Chroma's
default embedding model. It needs network access once; after that the harness
runs offline against the frozen snapshots. (In a fully air-gapped environment,
configure Chroma with a local embedding function.)

### Testing the scorer itself (offline, no key)

`test_harness_scoring.py` verifies the scoring logic against crafted
Answer/case pairs — a pass case plus every failure branch — with no LLM and no
network:

    PYTHONPATH=. python eval/test_harness_scoring.py

This guards the scorer from regressions independent of the agent.

## Script strata (which entry point to use)

The eval directory has grown across versions; the CURRENT pipeline is:

    acquire_data.py -> make_snapshots.py -> build_eval_from_manifest.py
    -> run_eval.py (--arm agent / --arm vanilla) -> compare_arms.py

plus `build_options_eval.py` (synthetic options set),
`build_attribution_eval.py` (synthetic attribution set, v11.6) and
`live_options_smoketest.py` (unscored live check, v11.3+).

Earlier-strata scripts kept for reproducibility of past results, not for new
runs: `build_golden.py` / `validate_golden.py` (superseded by the manifest
pipeline), `run_aapl_eval.py` / `build_aapl.py` (superseded by run_eval.py on
manifest-built sets), and the retrieval-debugging trio
(`diagnose_retrieval.py`, `analyze_retrieval_probes.py`, `run_lookahead.py`)
which are diagnostic tools, not eval entry points.
