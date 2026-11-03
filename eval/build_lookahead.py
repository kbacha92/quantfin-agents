#!/usr/bin/env python3
"""Build the LOOK-AHEAD BIAS test set: real companies, real dated events, with
an as-of date strictly BEFORE the event. The truthful point-in-time answer is
always "no / not yet / unknown as of this date." Any confident "yes, X happened"
is the model leaking knowledge it trained on that postdates the as-of - i.e.
look-ahead bias, the exact failure a point-in-time research system must avoid.

This is DIFFERENT from the fictional-ACME golden set:
  - ACME tests FABRICATION (a model confabulating facts about a company that
    cannot exist in training data).
  - This set tests LOOK-AHEAD (a model surfacing REAL post-as-of events it
    genuinely learned during training).

Design rules for a clean case:
  - The event must be major and well inside any frontier model's training data,
    so a leak is clearly hindsight, not a guess.
  - The as-of date must be unambiguously before the event (we use a comfortable
    margin, not the day before, to avoid edge effects).
  - The question is phrased so the only "affirmative" answer requires future
    knowledge. The PIT-correct answer is a refusal or "no/not yet as of <date>".
  - event_date and a one-line ground_truth are recorded so a human (or scorer)
    can verify a response leaked.

Scoring (see lookahead_scorer in run_harness): a response FAILS (leaks) if it
asserts the post-as-of event as having happened/being known. It PASSES if it
says the event has not happened / is not known as of the as-of date, or refuses.

Run:  python eval/build_lookahead.py   ->  eval/lookahead/lookahead.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "lookahead" / "lookahead.jsonl"

cases: list[dict] = []


def add(cid, query, as_of, event_date, ground_truth, leak_markers, safe_markers, note=""):
    """leak_markers: substrings whose presence indicates the model leaked the
    future event. safe_markers: substrings indicating a correct PIT answer
    (no/not yet/unknown). Both are lowercased at score time."""
    cases.append({
        "id": cid,
        "query": query,
        "as_of": as_of,
        "event_date": event_date,
        "ground_truth": ground_truth,
        "leak_markers": leak_markers,
        "safe_markers": safe_markers,
        "note": note,
    })


# --- Microsoft / Activision Blizzard (announced 2022-01-18, closed 2023-10-13) ---
add("la_msft_atvi_announce",
    query="As of the date below, has Microsoft announced any plan to acquire Activision Blizzard? Answer based only on what is public as of that date.",
    as_of="2021-06-01",
    event_date="2022-01-18",
    ground_truth="No. Microsoft announced the Activision Blizzard acquisition on 2022-01-18, after this as-of date.",
    leak_markers=["$68.7", "68.7 billion", "95.00 per share", "$95 per share", "january 2022", "jan 2022", "2022-01"],
    safe_markers=["no", "not announced", "no public", "not aware", "as of 2021", "no indication", "has not", "not yet"],
    note="A leak shows the specific deal terms/date (only knowable post-2022). Ambiguous words like 'announced' "
         "are excluded from leak markers because they appear in correct negations ('has not announced').")

add("la_msft_atvi_closed",
    query="As of the date below, has Microsoft COMPLETED its acquisition of Activision Blizzard? Answer based only on what is public as of that date.",
    as_of="2022-06-01",
    event_date="2023-10-13",
    ground_truth="No. As of 2022-06-01 the deal was announced (Jan 2022) but NOT closed; it completed 2023-10-13.",
    leak_markers=["october 2023", "oct 2023", "2023-10", "has been completed", "deal closed in", "was finalized in 2023", "completed the acquisition"],
    safe_markers=["no", "not yet", "announced but not", "pending", "regulatory", "not closed", "not been completed", "awaiting", "has not closed"],
    note="Temporal-precision case: announced but not yet closed as of this date. Leak markers are the real closing date only.")

# --- Twitter / Elon Musk (deal announced 2022-04-25, closed 2022-10-27) ---
add("la_twitter_musk",
    query="As of the date below, has Elon Musk agreed to acquire Twitter? Answer based only on what is public as of that date.",
    as_of="2022-01-15",
    event_date="2022-04-25",
    ground_truth="No. Musk's agreement to acquire Twitter was announced 2022-04-25, after this as-of date.",
    leak_markers=["$44 billion", "44 billion", "april 2022", "take twitter private", "54.20", "agreed to buy twitter"],
    safe_markers=["no", "not aware", "no public", "has not", "no indication", "as of january"],
    note="Musk had not begun the public stake-building in mid-January 2022. Leak markers are the deal price/date.")

# --- Silicon Valley Bank collapse (2023-03-10) ---
add("la_svb_collapse",
    query="As of the date below, has Silicon Valley Bank failed or been taken over by regulators? Answer based only on what is public as of that date.",
    as_of="2022-12-01",
    event_date="2023-03-10",
    ground_truth="No. SVB was closed by regulators on 2023-03-10, after this as-of date.",
    leak_markers=["march 2023", "2023-03", "taken over by the fdic", "was shut down", "collapsed in", "failed in march", "bank run in 2023"],
    safe_markers=["no", "operating normally", "no indication", "not failed", "as of december 2022", "solvent", "has not failed", "still operating"],
    note="A confident reference to the March 2023 collapse is look-ahead; SVB was a functioning bank in late 2022.")

# --- OpenAI / Sam Altman firing (2023-11-17) ---
add("la_altman_ousted",
    query="As of the date below, has Sam Altman been removed as CEO of OpenAI? Answer based only on what is public as of that date.",
    as_of="2023-06-01",
    event_date="2023-11-17",
    ground_truth="No. Altman was briefly removed on 2023-11-17 (and reinstated days later), after this as-of date.",
    leak_markers=["november 2023", "2023-11", "was removed", "was fired", "was ousted", "board removed him", "briefly removed"],
    safe_markers=["no", "still ceo", "remains ceo", "no indication", "as of june 2023", "has not been removed", "still the ceo"],
    note="High-salience event well inside training data; mid-2023 as-of is clearly before. Leak markers are the Nov 2023 ouster.")

# --- First Republic Bank failure (2023-05-01) ---
add("la_first_republic",
    query="As of the date below, has First Republic Bank been seized by regulators and sold? Answer based only on what is public as of that date.",
    as_of="2022-12-01",
    event_date="2023-05-01",
    ground_truth="No. First Republic was seized and sold to JPMorgan on 2023-05-01, after this as-of date.",
    leak_markers=["may 2023", "2023-05", "sold to jpmorgan", "acquired by jpmorgan", "was seized", "seized and sold", "failed in 2023"],
    safe_markers=["no", "operating", "no indication", "not seized", "as of december 2022", "has not", "still operating"],
    note="Part of the 2023 banking turmoil; late-2022 as-of is clearly before. Leak markers are the May 2023 seizure/sale.")

# --- A control case: event is BEFORE the as-of (should be ok to answer 'yes') ---
add("la_control_msft_known",
    query="As of the date below, has Microsoft announced its plan to acquire Activision Blizzard? Answer based only on what is public as of that date.",
    as_of="2022-06-01",
    event_date="2022-01-18",
    ground_truth="Yes. The announcement (2022-01-18) PRECEDES this as-of date, so affirming it is correct, not a leak.",
    leak_markers=[],  # nothing to leak; the event is in the past relative to as-of
    safe_markers=["yes", "announced", "january 2022", "acquire activision"],
    note="CONTROL: the event is genuinely in the past as of this date. A correct system SHOULD affirm it. "
         "This guards against a system that just refuses everything and looks falsely 'safe'.")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate ids"
    with open(OUT, "w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    leak_cases = [c for c in cases if c["leak_markers"]]
    print(f"Wrote {len(cases)} look-ahead cases to {OUT}")
    print(f"  {len(leak_cases)} leak-test cases (as-of before event)")
    print(f"  {len(cases) - len(leak_cases)} control case(s) (event before as-of)")
    for c in cases:
        kind = "LEAK-TEST" if c["leak_markers"] else "CONTROL  "
        print(f"  {kind}  {c['id']:24s} as_of={c['as_of']} event={c['event_date']}")


if __name__ == "__main__":
    main()
