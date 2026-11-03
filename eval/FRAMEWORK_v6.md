# Automated Evaluation Framework (v6)

v6 adds a data-acquisition pipeline so you can stand up a full guarded-vs-vanilla
evaluation for ANY ticker from real data, with the point-in-time as-of logic
derived automatically from actual SEC filing dates - eliminating the manual
date-verification that was the most error-prone step.

## The pipeline (four commands)

    # 1. Acquire real data: 10-Ks from EDGAR + prices from yfinance
    export SEC_IDENTITY="Your Name your@email.com"     # SEC fair-access policy
    PYTHONPATH=. python eval/acquire_data.py --ticker AAPL --start 2020-12-31 --end 2024-12-31
    #   -> data/docs/<date>_AAPL_10-K.txt  (indexed into the corpus)
    #   -> eval/raw_prices/AAPL.csv         (full daily history)
    #   -> eval/manifest/AAPL.json          (real filing dates + price coverage)

    # 2. Build cases anchored to the REAL filing dates
    PYTHONPATH=. python eval/build_eval_from_manifest.py --ticker AAPL
    #   -> eval/aapl/aapl.jsonl

    # 3. Generate the price snapshots the cases need (names derived automatically)
    PYTHONPATH=. python eval/make_snapshots.py --ticker AAPL --from-eval eval/aapl/aapl.jsonl
    #   -> data/api_cache/yfinance_AAPL_<asof>_<lookback>d.csv

    # 4. Run both arms and compare
    PYTHONPATH=. python eval/run_eval.py --dataset eval/aapl/aapl.jsonl --arm agent   --repeat 3 --report eval/aapl_agent.json
    PYTHONPATH=. python eval/run_eval.py --dataset eval/aapl/aapl.jsonl --arm vanilla --repeat 3 --report eval/aapl_vanilla.json
    PYTHONPATH=. python eval/compare_arms.py eval/aapl_agent.json eval/aapl_vanilla.json

## What the framework tests (the two goals)

1. **Do the guardrails hold?** The PIT cases are anchored to real filing dates.
   For each 10-K, a "future-invisible" case sets the as-of one day BEFORE the
   filing and asks about its contents - the guarded agent must return
   insufficient_data and must NOT cite the not-yet-filed document. A
   "visible" case sets the as-of after the filing and expects it to be citable.

2. **Does the agent outcompete a vanilla model with the same tools?** The
   `--arm vanilla` runs the same tools with guardrails OFF (retrieval has no
   as-of date filter, no ticker gate, no PIT evaluator). The comparison isolates
   the guardrails: same model, same tools, same corpus.

## Case readiness (what's automatic vs what you still verify)

`build_eval_from_manifest.py` tags each case:

- **READY** - PIT future-invisible cases. The as-of/forbid logic is correct BY
  CONSTRUCTION from the real filing date; runnable as-is, no verification needed.
- **VERIFY_VISIBLE** - PIT visible cases. Confirm the chosen filing really is the
  "most recent" as of the after-date (i.e. no later 10-K slipped in between).
- **VERIFY_VALUE** - numerical + provenance cases. The as-of and snapshot naming
  are automatic, but you must fill ONE figure: run the tool once on the generated
  snapshot to read the exact RSI/drawdown/vol, or read the revenue figure from
  the filing. Until filled, the runner skips these (they still contain a [VERIFY]).

So the manual work is reduced to: confirming "most recent" on the visible cases,
and pasting in a handful of exact numbers. The date logic - where mistakes
silently invert a PIT test - is no longer hand-done.

## Important notes

- **Network:** `acquire_data.py` needs SEC EDGAR + yfinance access and is the
  only networked step. Set `SEC_IDENTITY` or SEC may rate-limit you. The agent
  runs offline afterward (offline-strict during eval).
- **The future filing must be in the corpus for the vanilla leak to be real.**
  Because acquisition pulls every 10-K in the range INTO the corpus, the vanilla
  arm (no date filter) genuinely can retrieve a post-as-of filing - so the PIT
  comparison tests retrieval leakage, not just model memory. This is the point.
- **Repeat always.** Both arms are nondeterministic; compare pass-rates, and
  expect borderline cases to be FLAKY.
- **Read per-capability.** The story is "vanilla leaks future filings / miscomputes;
  the agent refuses and computes exactly," not a single aggregate.
- **One 10-K is not enough for a rich PIT test.** A range that yields several
  annual filings (e.g. 4-5 years) gives multiple future/visible boundaries.

## v7.1 — shared corpus index (performance)

Re-ingesting full 10-Ks into a fresh Chroma index for every case made agent runs
very slow (minutes per case; an hours-long full run on real filings). v7.1 builds
the corpus index ONCE and reuses it across all cases:

- `run_eval.py` prebuilds a shared index before the case loop, then points every
  case's data dir at it (`QUANT_EVAL_SKIP_INGEST=1` makes `ingest()` a no-op so
  the prebuilt index is reused, not re-embedded).
- `_index_folder` now skips chunks already present in the collection, so even a
  direct re-ingest is near-instant instead of re-embedding everything.

Correctness is unchanged: the point-in-time filter is applied at QUERY time
(`pub_date <= as_of`), so one shared index over all filings still gives each case
the correct as-of-filtered view. The guardrail behaviour is identical; only the
indexing cost moved from per-case to once-per-run.

Disable with `--no-shared-index` to restore full per-case isolation (slower).

## v8.1 — pluggable retrieval backends (scaffolded demonstration)

v8.1 adds three retrieval backends behind a common interface so the eval can
compare AGENT TYPES (by how they retrieve facts) on the same cases:

  - rag    : semantic vector search (existing; nondeterministic - the flaky cluster)
  - sql    : exact queries over a SQLite facts table (deterministic)
  - graph  : 1-hop traversal over an in-memory fact graph (deterministic)

All three return the same Evidence shape and apply the same point-in-time filter
(pub_date <= as_of), so they are drop-in interchangeable and the guardrail is
identical regardless of backend. Build the stores and compare:

    python eval/build_fact_stores.py        # -> data/facts.db, data/facts_graph.json
    python eval/compare_backends.py --repeat 8   # determinism per backend, no API key

Verified: sql + graph are byte-identical across repeated runs (deterministic),
and the PIT filter holds on all three (e.g. the Beacon acquisition dated
2025-05-20 is invisible at as_of 2025-03-01, visible at 2025-06-01).

SCOPE / HONESTY: these are REAL working skeletons, not mocks - but
demonstration-scale. The SQL and graph stores are populated from a shared curated
fact seed (eval/build_fact_stores.py), NOT by parsing full 10-K prose. That
extraction (prose -> rows / prose -> triples) is the large build deliberately
deferred; the point of the scaffold is to prove the eval can compare backends,
with each backend genuinely functioning over a shared fact set. Wiring a backend
selector into the full agent loop (so golden cases run end-to-end per backend) is
the natural next step beyond this scaffold.

## v8.2 — cost reporting, trace dump, pit_05 assertion fix

Three additions prompted by diagnosing the golden run:

1. **Per-case token + cost reporting.** llm.complete() now captures resp.usage
   (previously discarded). The harness prints, per case, input tokens / output
   tokens / USD cost, and a run-wide total + per-case average. Token counts are
   exact; the dollar figure uses config.PRICING_PER_MTOK (verify current rates
   at anthropic.com/pricing). Reports include a "cost" block.

2. **Trace inspection.** --show-answer prints each case's answer text, status,
   and citations (the diagnostic that was missing when combo_01 returned
   insufficient_data with no way to see why). --save-trace DIR copies each
   case's trace JSONL out of the temp workspace for inspection.

3. **pit_05 assertion fix.** must_not_contain previously listed "acquisition of
   Beacon", which a CORRECT denial ("no acquisition of Beacon appears...")
   trips - a false positive, the same class of bug fixed earlier for combo_02.
   Now it forbids only AFFIRMATIVE leaks ("acquired Beacon", "completed the
   acquisition", "800 million"). Verified: a correct denial passes, an
   affirmative leak still fails.

KNOWN, NOT BLINDLY PATCHED:
- pit_05 also sometimes returns insufficient_data instead of ok (the existential
  "has X happened" judgment). Left as a real signal, not masked by loosening the
  status check.
- The flaky cluster (pit_06, pit_09, prov_02, pit_11) is retrieval
  nondeterminism (got=[] - retrieval occasionally returns nothing). The real fix
  is the deterministic SQL backend (v8.1), not assertion tweaks - wiring the
  agent loop to use it per-backend is the next step. Use --show-answer to
  confirm these are empty-retrieval, not reasoning, failures.
- combo_01 returns insufficient_data deterministically; use --show-answer to see
  the agent's reasoning before deciding whether it is the agent, the case
  expectation, or the iteration budget.

## v9.1 — fixes from the --show-answer diagnosis

The v8.2 trace dump let us SEE why the failing cases failed. Three findings, two
fixed in code, the rest documented honestly:

1. **combo_01 (real agent bug, FIXED).** The trace showed the agent COMPUTING the
   RSI ("Found: quant:rsi:SYN:14:2025-03-14") and then declaring that same RSI
   "still missing" - looping until the iteration cap and burning ~$0.43/run. The
   evaluator wasn't crediting a computed quant evidence item as satisfying the
   numeric sub-question. Fixed in prompts.EVAL_SYSTEM: a computed quant/price
   evidence item now explicitly counts as the answer to the numeric part; the
   evaluator must not re-request a value already in evidence.

2. **pit_05 (assertion bug, FIXED properly this time).** v8.2 forbade "acquired
   Beacon" / "has acquired Beacon" - but a CORRECT denial ("has NOT acquired
   Beacon") contains those substrings, so the fix flipped the false positive
   rather than removing it. Substring matching cannot distinguish "has acquired"
   from "has NOT acquired". v9.1 forbids only deal specifics a denial never
   states (the $800M price, explicit "completed the acquisition"). Verified
   against the ACTUAL failing answer from the trace: it now passes; a real leak
   still fails. The PIT guardrail remains enforced by forbid_citation_source_ids.

NOT PATCHED (deliberately, to avoid unverifiable changes):
- pit_06 (4/5) and pit_11 (2/5) are retrieval nondeterminism (got=[] - retrieval
  occasionally returns nothing; pit_11's trace literally shows "Found: nothing").
  When retrieval works, the agent's answers are excellent (see pit_06's 4/5 run).
  The real fix is the deterministic SQL backend (v8.1), not assertion or loop
  tweaks. pit_11 also exposes the stranded early-stop giving up at 3 iterations on
  a two-document case; tightening that is a core-orchestrator change that cannot
  be verified without live LLM calls, so it is left for a tested change rather
  than shipped blind.

## v9.2 — retrieval instrumentation (settle infra vs inherent flakiness)

The flaky cluster all carries the got=[] signature (retrieval returns NOTHING).
Two hypotheses: a cold-start readiness race (index queried before warm) vs
inherent semantic-ranking variance (populated index, query just returns nothing).
The fix differs hugely (warm/retry vs a routing rebuild), so v9.2 instruments
retrieval to settle it with data instead of priors.

Every retrieval now emits a `retrieval_probe` trace record capturing, at the
instant of the result: the query's position in the process (query_index), the
collection.count() at that moment, whether a retry was needed, and the outcome
(hit / empty_correct / empty_despite_populated). The orchestrator drains these
into the trace after each tool call.

How to run the diagnosis:
    # run the flaky cluster with traces saved
    PYTHONPATH=. python3 eval/run_harness.py --ids pit_06_acquisition_visible_later,\
pit_09_guidance_withdrawn,pit_11_two_docs_visible,pit_12_analyst_note_visible \
      --repeat 20 --save-trace eval/probe_traces
    # analyze
    PYTHONPATH=. python3 eval/analyze_retrieval_probes.py eval/probe_traces

The analyzer prints two decisive distributions:
  [1] empties by QUERY POSITION  - front-loaded => race; scattered => inherent
  [2] COLLECTION COUNT at emptiness - <=0 => not-ready (race); >0 => variance
  [3] retry effect - retries resolving empties => race; empty_despite_populated => inherent

This adds only logging plus the count/retry bookkeeping the retry path already
did; it changes no retrieval behaviour. Verified against simulated race,
inherent, and empty-collection scenarios: each is recorded distinctly.

## v9.3 — fix trace capture so the probe diagnosis gets all runs

The v9.2 probe run captured only 4 trace files for 80 runs: --save-trace copied
the trace OUTSIDE the repeat loop (using only last_res) and named files by case id
alone, so each case's 20 repeats collapsed to one file - and the analyzer saw a
handful of (coincidentally clean) runs. The probe instrumentation was fine; the
capture was lossy.

Fix: --save-trace now copies each repeat's trace INSIDE the loop, named
{cid}__rep{NN}__{traceid}.jsonl, so all N runs are kept. (Each repeat already has
a unique uuid trace_id, so there was never content collision - only the harness's
save step dropped them.) Re-run the v9.2 diagnosis command and the analyzer will
now see every run's probes, including the minority that fail with got=[].

## v10.1 — plan-validation repair (the actual root cause of the flaky cluster)

The instrumentation chain (v9.2 probes -> v9.3 full capture) DISPROVED the
retrieval-flakiness hypothesis: 126/126 retrievals were hits, zero empties. The
trace then showed the real cause - on failing runs the planner emitted
doc_search with NO query (args={}), which crashed the call
("missing required argument 'query'"); with edgar_fetch also failing (ACME is
fictional, correctly not on EDGAR), the run had no working retrieval path,
stranded, and returned insufficient_data. The got=[] in scores was never
retrieval returning empty - it was retrieval never running.

Fix: _repair_action validates each action before execution. A retrieval tool
(doc_search/knowledge_search) emitted without a query gets the user's question
injected as the query, and the repair is logged ('plan_repair') so the planner's
omission rate stays measurable. This converts the crash-and-strand into a working
retrieval. Unit-tested (tests/test_plan_repair.py): query-less and blank-query
calls are repaired, real queries and non-retrieval tools are untouched, the
original action is not mutated.

This was found by INSTRUMENTING, not guessing - the diagnosis moved three times
(retrieval-empty -> tool-error -> planner-drops-argument), each step ruled out by
trace data rather than assumption. Worth remembering: the "RAG is flaky, needs a
SQL backend / router" thread was aimed at a layer that was never the problem.

REMAINING (honest): the planner still sometimes OMITS the doc_search action
entirely from its initial plan (reasoning about it in 'thoughts' but not emitting
it in 'actions'), only attempting it later under recovery. The repair catches the
malformed call when it eventually fires; a prompt-level fix so doc_search is
reliably planned up front is a separate, behavioral change best verified on a live
run (raise it as v10.2 if the repair alone doesn't lift the flaky cases to clean).

## v11.1 — options pricing tools (Phase 1 of the options eval)

The options eval needs the tools to exist first. v11.1 adds the verified
computational core (the eval cases come next):

- options.py: Black-Scholes price, Greeks (delta/gamma/vega/theta/rho), and an
  implied-vol solver (Newton-Raphson + bisection fallback). European, closed-form.
- option_chain.py: synthetic chain generation from a KNOWN volatility smile, plus
  curve recovery (invert BS per strike). This is the eval ground-truth mechanism -
  build prices from a chosen smile, the agent must recover it.
- option_tool.py: OptionTool (option_calc) wraps price/greeks/implied_vol as an
  agent-callable tool returning Evidence; registered in REGISTRY.

Verified (tests/test_options.py, 7 tests): textbook benchmark S=K=100,T=1,r=5%,
sigma=20% -> call 10.4506 / put 5.5735 exactly; put-call parity; Greeks match
known values; IV round-trips to 1e-4 for calls and puts; arbitrage-priced inputs
are rejected; synthetic chain recovers its smile to 1e-7.

WHY SYNTHETIC (decision, recorded): yfinance gives only the CURRENT chain, not
historical/as-of chains, so a live options eval cannot be reproducible or scored
against known answers - it breaks the point-in-time basis the rest of the eval
relies on. The scored eval therefore uses synthetic chains with analytic ground
truth (mirrors the SYN/DDN approach); a live yfinance download is a separate,
UNSCORED smoke test.

STILL TO BUILD (next versions): (1) synthetic-chain eval cases scoring the agent's
IV recovery / vol-curve construction against the known smile; (2) the live
yfinance smoke test for the download path; (3) optionally a vol-curve chart via
the existing chart tool.

## v11.2 — synthetic options eval (agent vs vanilla)

The scored options eval, built on the v11.1 tools. Cases are generated with
ground truth computed by the real tools from a known volatility smile - no
[VERIFY] placeholders, self-consistent by construction.

    python eval/build_options_eval.py     # -> eval/options/options.jsonl (5 cases)

    # both arms (same shape as the AAPL eval)
    export ANTHROPIC_API_KEY=$(python3 -c "import keys; keys.apply(); import os; print(os.environ['ANTHROPIC_API_KEY'])")
    PYTHONPATH=. python3 eval/run_eval.py --dataset eval/options/options.jsonl --arm agent   --repeat 3 --report eval/options_agent.json
    PYTHONPATH=. python3 eval/run_eval.py --dataset eval/options/options.jsonl --arm vanilla --repeat 3 --report eval/options_vanilla.json
    PYTHONPATH=. python3 eval/compare_arms.py eval/options_agent.json eval/options_vanilla.json

Cases: BS call price, call delta (Greeks), implied-vol solving, a vol-smile wing
IV, and a put price. The implied-vol and vol-curve cases are the decisive
agent-vs-vanilla contrast: solving implied vol requires inverting Black-Scholes
(the option_calc tool does Newton-Raphson); a vanilla model guessing by hand
misses the tolerance. Options cases need no snapshots or filings - pure
computation - so staging is trivial vs the AAPL eval.

Verified (tests/test_options_eval.py): every case's expected_value equals what
the tools independently compute; scorer passes a correct answer and fails a wrong
one.

STILL TO BUILD: the live-yfinance download smoke test (unscored; verify on a
machine with network - the sandbox can't reach Yahoo).

## v11.3 — live yfinance options smoke test (unscored)

The separate, unscored counterpart to the synthetic eval - verifies the live
download + compute path on real market data.

    python eval/live_options_smoketest.py --ticker AAPL
    python eval/live_options_smoketest.py --ticker MSFT --max-expiries 3 --moneyness 0.08

Downloads the CURRENT chain via yfinance, solves implied vol from real mid-prices
across near-the-money strikes, and reports a vol curve per expiry with sanity
checks (IV finite and in-bounds, ATM IV plausible, >=3 points for a smile). NaN/
illiquid quotes are skipped and reported, not failed.

NOT SCORED, by design: live chains are current-only (not reproducible) and have
no frozen ground truth. The reproducible scored eval remains the synthetic
options.jsonl. This is the "both" split: synthetic = scored correctness; live =
unscored capability check.

VERIFICATION CAVEAT: the download-independent logic (year-fraction, IV-solving on
realistic quotes) is unit-verified; the actual yfinance call is NOT tested here
because the build sandbox has no network to Yahoo. Run on a networked machine;
first-run watch items: yfinance installed, Yahoo reachable, and some tickers
return NaN bid/ask on illiquid strikes (expected - those are skipped).

## v11.5 — equity attribution tool (NLPHFEA port)

Ports the deterministic attribution ENGINE from the NLPHFEA framework as an
agent tool (equity_attribution), not its multi-agent loop (the host system
already is one). Ops: list / profile / brinson_fachler / cooper_li over
data/portfolios/ files, PIT-gated by filename date exactly like data/docs/.

Verified (tests/test_attribution.py, 10 tests): hand-computed Brinson-Fachler
and Cooper-Li cases; both reconciliation identities to 1e-12; NLPHFEA's own
example portfolios reproduce their shipped characteristics references ($1,598
and $6,377 at $500k); percent/dollar-string coercion; PIT gating (post-as-of
file refused as lookahead, hidden from list); path-traversal rejection;
deterministic source_ids.

Eval note: attribution cases are naturally golden-set material - a synthetic
portfolio with chosen weights/returns has exact analytic attribution, same
discipline as the options eval. Built in v11.6 - see below.

## v11.6 — scored attribution eval (agent vs vanilla)

The scored eval for equity_attribution, mirroring the options eval design.

    PYTHONPATH=. python3 eval/build_attribution_eval.py
    PYTHONPATH=. python3 eval/run_eval.py --dataset eval/attribution/attribution.jsonl --arm agent   --repeat 3 --report eval/attr_agent.json
    PYTHONPATH=. python3 eval/run_eval.py --dataset eval/attribution/attribution.jsonl --arm vanilla --repeat 3 --report eval/attr_vanilla.json
    PYTHONPATH=. python3 eval/compare_arms.py eval/attr_agent.json eval/attr_vanilla.json

Design points:
- Ground truth computed by the REAL engine from a fixed 9-name, 3-sector
  long/short synthetic book (non-round weights/returns so mental arithmetic
  misses the tolerances). No [VERIFY] placeholders.
- The holdings table is EMBEDDED in each question, so the vanilla arm has
  identical information - the comparison isolates tool + guardrails, not data
  access. Required attribution:* citations are producible only by the tool.
- attr_cl_sse is the decisive contrast (short-sign conventions); the scorer
  test proves a sign-flipped answer fails.
- attr_pit_lookahead names a post-as-of portfolio file WITHOUT embedding its
  numbers: the guarded agent's tool refuses ('lookahead'), expected
  insufficient_data; forbidden citations + the future P&L in four number
  formats (comma/no-comma x 2dp/rounded - single-format substring guards are
  brittle, a lesson re-learned from pit_05) catch a fabricating leak.
- run_eval.py stages case-declared 'portfolios' files into the workspace's
  portfolios/ dir (both shared-index and isolated branches), mirroring
  snapshot staging; the tool's own PIT gate then decides visibility.

Verified (tests/test_attribution_eval.py, 6 tests): case values match
independent engine recomputation; the future book's P&L differs from the
visible book's by >$1k (the leak is detectable); the scorer passes a correct
cited answer and fails wrong-sign, uncited, and lookahead-leak answers; the
tool run end-to-end on the staged book emits the exact expected source_id and
refuses the future file. Live both-arms runs happen on your machine, as always.

## v11.6.1 — scorer sign-extraction fix (live-run finding)

The first live run of the attribution eval (agent arm: 2 PASS, 2 FLAKY,
2 FAIL) was diagnosed from --show-answer trace evidence: every failure was a
scorer artifact, not an agent error. The report's tell was a perfect sign
correlation - the only positive-valued case passed 3/3; the negative dollar
cases failed 0/6. Answer text (repr) showed correct values signed with an en
dash (U+2013) or minus sign (U+2212) before '$', both invisible to the old
ASCII-adjacent regex. Fix: normalize dash variants and currency-separated
signs before extraction, strictly preserving sign semantics. The verbatim
failing production answers are frozen as regression tests
(tests/test_scorer_sign_extraction.py). Diagnosis chain: report pattern ->
mechanism demonstrated offline -> confirmed with live answer text -> fix +
regressions. Instrument, don't guess.

## v11.7 — full Cooper-Li model, case generator, eval statistics

Engine: cooper_li_full implements eqn (11) with trades (MM, eqn 8) and ETFs
(Chart-6 constituent decomposition + AE, eqn 9B); reduces exactly to the base
decomposition without companions, so all v11.5/v11.6 ground truth is
untouched. Paper-exact: tests/test_attribution_full.py pins every number in
Charts 1-7 of Cooper & Li (2012), the same external-reference discipline as
Hull for the options pricer.

Generated population: eval/generate_attribution_cases.py --n 30 --seed 42
emits a seeded case population (plain/mm/etf/both) for statistically
defensible arm comparisons; run both arms on eval/attribution/generated_42.jsonl
exactly like the hand-crafted set (the harness stages all companion files a
case names in 'portfolios').

Statistics: compare_arms now reports, per-CASE (never per-run): mean pass
rates, paired bootstrap 95% CI on the delta (seeded, deterministic), exact
McNemar p on discordant cases, and a per-capability table. The hand-crafted
six-case set remains the fixed regression bench; the generated population is
where inference happens.
