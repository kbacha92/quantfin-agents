# Look-ahead bias test — agent vs vanilla frontier model

This set tests the agent's core reason to exist: a vanilla frontier model has
real future events baked into its weights and tends to **leak** them when asked
about a past as-of date. The point-in-time agent, with no corpus knowledge of
these events and as-of enforcement, should refuse or answer "not as of this
date."

This is different from the fictional-ACME golden set:
- **ACME golden set → fabrication test.** A model confabulating facts about a
  company that cannot be in training data. Tests groundedness.
- **This set → look-ahead test.** A model surfacing REAL post-as-of events it
  genuinely learned in training. Tests point-in-time integrity against true
  hindsight.

## The cases (`eval/lookahead/lookahead.jsonl`)

Each case is a real company + real dated event + an as-of date strictly before
the event. The truthful answer is always "no / not yet / unknown as of this
date." 6 leak-test cases (e.g. Microsoft–Activision announcement, Musk–Twitter,
SVB collapse, Altman ouster, First Republic) plus **1 control** where the event
*precedes* the as-of — the control catches a system that just refuses everything
and looks falsely "safe."

Regenerate / edit:

    python eval/build_lookahead.py

Each case records `event_date`, a one-line `ground_truth`, and `leak_markers` /
`safe_markers`. Leak markers are deliberately **specific post-as-of details**
(exact dates, dollar amounts, named acquirers) that cannot appear in a correct
negation — so "has **not** announced" doesn't false-trigger on the word
"announced."

## Running it (harness — for the rate)

    export ANTHROPIC_API_KEY=...
    # agent arm: should be SAFE on every leak-test case
    PYTHONPATH=. python eval/run_lookahead.py --arm agent --repeat 3 --report eval/la_agent.json
    # vanilla arm: expected to LEAK on several
    PYTHONPATH=. python eval/run_lookahead.py --arm baseline --repeat 3 --report eval/la_vanilla.json

Each case is scored SAFE / LEAK / UNCLEAR by marker detection. The runner stores
the full answer text in the report so you can **audit any verdict** — especially
UNCLEAR ones, where the markers didn't fire cleanly and a human should read it.

Important honesty notes:
- The leak scorer is marker-based and transparent, not a judge. Treat LEAK/SAFE
  as a strong signal, but read the `sample_text` for anything ambiguous.
- The vanilla arm is given a system prompt that explicitly tells it to answer
  only as of the date and to say so if something hadn't happened. So a leak is
  the model failing *despite* being asked not to — which is the honest, strong
  form of the test (we are not tricking it).

## Running it (browser — for the screenshots)

The harness gives you the rate; the browser gives you the persuasive artifact.
For a stakeholder demo, paste a leak-test prompt verbatim into ChatGPT / Claude
/ your Streamlit agent and screenshot the contrast:

    "As of 2021-06-01, has Microsoft announced any plan to acquire Activision
    Blizzard? Answer based only on what is public as of that date."

A vanilla model often answers "yes, in January 2022 for $68.7B" — leaking the
future. Your agent answers "no public indication as of this date."

Caveats for the browser demo (state these if you present it):
- A single browser run is ONE sample; frontier models are nondeterministic and
  are patched over time, so a model that leaks today may refuse next month. Use
  screenshots as "here is a real instance of the failure," not a rate.
- For a *rate*, use the harness with `--repeat` and an API model — that's
  reproducible; a browser session is not.
- The browser vanilla model may also have live web access, which is a different
  capability from training-data look-ahead; note which one you're demonstrating.

## What a good result looks like

- **Agent:** SAFE on all 6 leak-test cases, SAFE (correctly affirms) on the
  control. If the agent leaks, that's a serious finding — it shouldn't have
  these events in any corpus, so a leak means the model's own knowledge is
  bleeding through the synthesis layer despite the as-of framing.
- **Vanilla:** LEAK on several leak-test cases = the demonstration that
  motivated the whole point-in-time architecture. The control should still be
  SAFE for both (it's a legitimately answerable past event).
