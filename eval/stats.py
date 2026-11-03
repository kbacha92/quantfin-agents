"""Statistics for arm comparisons (v11.7).

The unit of analysis is the CASE, not the run: repeats of one case share its
difficulty, so run-level counting overstates the effective sample
(pseudo-replication). Everything here aggregates per-case first.

Three tools:
- per_case_rates(report)         case_id -> pass rate (passes/runs)
- bootstrap_delta_ci(a, v, ...)  paired bootstrap CI for mean(agent - vanilla)
                                 pass-rate difference, resampling CASES
- mcnemar_exact(b, c)            exact two-sided McNemar on discordant case
                                 counts (b = agent-only passes, c = vanilla-only),
                                 via the binomial(b+c, 0.5) distribution -
                                 no scipy dependency, exact by math.comb

All deterministic given the seed; no network, no LLM - fully testable offline.
"""
from __future__ import annotations

import math
import random


def per_case_rates(report: dict) -> dict[str, float]:
    out = {}
    for c in report.get("cases", []):
        runs = c.get("runs", 1) or 1
        out[c["id"]] = c.get("passes", 1 if c.get("status") == "PASS" else 0) / runs
    return out


def paired_rates(agent: dict, vanilla: dict) -> tuple[list[str], list[float], list[float]]:
    """Intersect case ids (order-stable) and return paired rate lists."""
    a_r, v_r = per_case_rates(agent), per_case_rates(vanilla)
    ids = [i for i in a_r if i in v_r]
    return ids, [a_r[i] for i in ids], [v_r[i] for i in ids]


def bootstrap_delta_ci(a_rates: list[float], v_rates: list[float],
                       n_boot: int = 10_000, seed: int = 0,
                       alpha: float = 0.05) -> tuple[float, float, float]:
    """(delta, lo, hi): mean paired difference with a percentile bootstrap CI.

    Resamples CASES with replacement (paired), preserving the per-case
    coupling between arms. Deterministic given seed.
    """
    if len(a_rates) != len(v_rates) or not a_rates:
        raise ValueError("need equal-length, non-empty paired rate lists")
    diffs = [a - v for a, v in zip(a_rates, v_rates)]
    n = len(diffs)
    delta = sum(diffs) / n
    rng = random.Random(seed)
    boots = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_boot)
    )
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return delta, lo, hi


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value on discordant counts.

    b = cases the agent passed and vanilla failed; c = the reverse.
    Under H0 (no difference) each discordant case is a fair coin:
    p = 2 * P[Binomial(b+c, .5) <= min(b, c)], capped at 1. Concordant cases
    carry no information and are correctly ignored.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def discordant_counts(a_rates: list[float], v_rates: list[float],
                      threshold: float = 1.0) -> tuple[int, int]:
    """Binarize per-case rates at `threshold` (default: all-runs-pass) and
    count discordant pairs for McNemar."""
    b = sum(1 for a, v in zip(a_rates, v_rates) if a >= threshold > v)
    c = sum(1 for a, v in zip(a_rates, v_rates) if v >= threshold > a)
    return b, c


def by_capability(agent: dict, vanilla: dict) -> dict[str, dict]:
    """Per-capability paired mean rates (cases grouped by 'capability')."""
    cap_of = {c["id"]: c.get("capability", "?") for c in agent.get("cases", [])}
    ids, a, v = paired_rates(agent, vanilla)
    out: dict[str, dict] = {}
    for i, ar, vr in zip(ids, a, v):
        g = out.setdefault(cap_of.get(i, "?"), {"n": 0, "a": 0.0, "v": 0.0})
        g["n"] += 1
        g["a"] += ar
        g["v"] += vr
    return {k: {"n": g["n"], "agent_rate": g["a"] / g["n"], "vanilla_rate": g["v"] / g["n"]}
            for k, g in out.items()}
