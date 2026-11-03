#!/usr/bin/env python3
"""Generate a seeded population of attribution eval cases (v11.7).

The statistical companion to build_attribution_eval.py: instead of six
hand-crafted cases, this samples N random synthetic portfolios and emits
scored cases whose ground truth is computed by the REAL engine (correct by
construction). This is what makes agent-vs-vanilla pass rates defensible as
population estimates rather than anecdotes about one hand-picked book.

    PYTHONPATH=. python3 eval/generate_attribution_cases.py --n 30 --seed 42

Case kinds (default mix, set with --mix plain,mm,etf,both):
  plain  0.4   holdings only; asks a base Cooper-Li quantity (PDR/SSE/NE)
  mm     0.2   + a trades companion (Cooper-Li eqn 8 market-making effect)
  etf    0.2   + ETF & constituent companions (Chart-6 decomposition + eqn 9B
               arbitrage effect; the ETF's implied exposures shift the base
               attribution, so 'both files matter')
  both   0.2   trades AND ETFs; asks for the full eqn-11 PDR

Every case embeds ALL its data (holdings + companions) in the question text so
the vanilla arm has identical information - the contrast isolates tool +
guardrails. Files are installed to data/portfolios/ with the shared date
prefix; the harness stages whatever the case's 'portfolios' list names.

Reproducibility: the seed is recorded in the dataset filename and in every
case id; the same seed yields byte-identical output (asserted in tests).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import pandas as pd

from quant_agent.tools.quant.attribution import cooper_li_full

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DIR = HERE / "attribution"
AS_OF = "2025-04-15"
DATE = "2025-03-31"
SECTORS = ["Alpha", "Beta", "Chi", "Delta"]


def _round6(x):
    return round(float(x), 6)


def gen_holdings(rng: random.Random) -> pd.DataFrame:
    n_sec = rng.randint(2, 4)
    sectors = SECTORS[:n_sec]
    rows = []
    k = 0
    for s in sectors:
        for _ in range(rng.randint(2, 4)):
            k += 1
            rows.append({
                "Ticker": f"S{k:02d}",
                "Sector": s,
                "Portfolio Weight": _round6(rng.uniform(-0.12, 0.18)),
                "Cap Weight": _round6(rng.uniform(0.02, 0.20)),
                "Return": _round6(rng.uniform(-0.09, 0.09)),
            })
    df = pd.DataFrame(rows)
    # Ensure at least one short (SSE questions need one) and nonzero universe.
    if not (df["Portfolio Weight"] < 0).any():
        df.loc[df.index[0], "Portfolio Weight"] = _round6(-abs(df.loc[df.index[0], "Portfolio Weight"]) - 0.01)
    return df


def gen_trades(rng: random.Random, holdings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in rng.sample(list(holdings["Ticker"]), k=rng.randint(1, 3)):
        px = round(rng.uniform(8, 120), 2)
        sh = rng.choice([-1, 1]) * rng.randint(50, 400)
        slip = round(rng.uniform(-0.4, 0.4), 2)
        rows.append({
            "Ticker": t,
            "Dollars": round(-sh * px, 2),          # signed: negative = bought
            "Shares": sh,
            "EndPrice": round(px + slip, 2),
        })
    return pd.DataFrame(rows)


def gen_etfs(rng: random.Random, holdings: pd.DataFrame):
    cons_tickers = rng.sample(list(holdings["Ticker"]), k=rng.randint(2, min(4, len(holdings))))
    beta = rng.choice([-2.0, -1.0, 1.0, 2.0])
    cons_rows = []
    for t in cons_tickers:
        start = round(rng.uniform(10, 80), 2)
        cons_rows.append({
            "ETF": "XF01", "Ticker": t,
            "SharesPerETF": rng.randint(1, 3),
            "StartPrice": start,
            "EndPrice": round(start * (1 + rng.uniform(-0.05, 0.05)), 2),
        })
    cons = pd.DataFrame(cons_rows)
    basket_start = float((cons["SharesPerETF"] * cons["StartPrice"]).sum())
    basket_end = float((cons["SharesPerETF"] * cons["EndPrice"]).sum())
    fair_move = beta * (basket_end - basket_start) / basket_start
    etf_start = round(rng.uniform(15, 60), 2)
    # ETF end deviates a little from fair -> a nonzero arbitrage effect
    etf_end = round(etf_start * (1 + fair_move + rng.uniform(-0.01, 0.01)), 2)
    etfs = pd.DataFrame([{"ETF": "XF01", "Beta": beta,
                          "SharesHeld": rng.randint(50, 300),
                          "StartPrice": etf_start, "EndPrice": etf_end}])
    return etfs, cons


def _csv_text(df: pd.DataFrame) -> str:
    return df.to_csv(index=False).strip()


def make_case(i: int, seed: int, kind: str, rng: random.Random, install_dir: Path):
    notional = float(rng.choice([100_000, 250_000, 500_000, 1_000_000]))
    holdings = gen_holdings(rng)
    trades = gen_trades(rng, holdings) if kind in ("mm", "both") else None
    etfs = cons = None
    if kind in ("etf", "both"):
        etfs, cons = gen_etfs(rng, holdings)

    grp, s = cooper_li_full(holdings, "Sector", "Portfolio Weight", "Cap Weight",
                            "Return", notional=notional,
                            trades=trades, etfs=etfs, constituents=cons)

    base = f"{DATE}_gen{seed}_{i:03d}"
    files = [f"{base}.csv"]
    holdings.to_csv(install_dir / files[0], index=False)
    blocks = [f"Holdings file {files[0]}:\n{_csv_text(holdings)}"]
    if trades is not None:
        fn = f"{base}_trades.csv"
        trades.to_csv(install_dir / fn, index=False)
        files.append(fn)
        blocks.append(f"Trades during the period ({fn}; Dollars signed, negative=bought):\n{_csv_text(trades)}")
    if etfs is not None:
        fn_e, fn_c = f"{base}_etfs.csv", f"{base}_etf_constituents.csv"
        etfs.to_csv(install_dir / fn_e, index=False)
        cons.to_csv(install_dir / fn_c, index=False)
        files += [fn_e, fn_c]
        blocks.append(f"ETF positions ({fn_e}):\n{_csv_text(etfs)}")
        blocks.append(f"ETF constituent baskets ({fn_c}):\n{_csv_text(cons)}")

    full = kind != "plain"
    sid = f"attribution:cl:{files[0]}:Sector:{notional:.4f}".rstrip("0").rstrip(".") + (":full" if full else "")
    data = "\n\n".join(blocks)
    preamble = (f"The portfolio store contains the following files (a long/short book "
                f"run at ${notional:,.0f} notional).\n\n{data}\n\n"
                "Using Cooper-Li attribution by Sector (universe = Cap Weight)")

    # Which quantity to ask for - the decisive per-kind contrasts.
    if kind == "plain":
        q_kind, expected = rng.choice([
            ("total P&L in dollars", s["PDR"]),
            ("total SHORT SELECTION effect (SSE) in dollars (watch the sign)", s["SSE_total"]),
            ("NET EXPOSURE effect (NE) in dollars", s["NE_total"]),
        ])
    elif kind == "mm":
        q_kind, expected = rng.choice([
            ("MARKET-MAKING effect (Cooper-Li eqn 8) in dollars", s["MM"]),
            ("FULL portfolio dollar return (PDR = base attribution + MM) in dollars", s["PDR"]),
        ])
    elif kind == "etf":
        q_kind, expected = rng.choice([
            ("ETF ARBITRAGE effect (eqn 9B) in dollars", s["AE"]),
            ("FULL portfolio dollar return (PDR, with the ETF decomposed into "
             "constituent exposures per the paper) in dollars", s["PDR"]),
        ])
    else:  # both
        q_kind, expected = ("FULL portfolio dollar return (PDR = base + market-making "
                            "+ ETF arbitrage, Cooper-Li eqn 11) in dollars", s["PDR"])

    return {
        "id": f"gen{seed}_{i:03d}_{kind}",
        "capability": f"attribution_{kind}",
        "as_of": AS_OF,
        "portfolios": files,
        "query": f"{preamble}, what is the {q_kind}? Use the equity_attribution tool.",
        "expected_status": "ok",
        "expect_citation_source_ids": [sid],
        "expected_value": round(expected, 2),
        "value_tolerance": max(5.0, abs(expected) * 0.005),
        "kind": kind,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mix", default="0.4,0.2,0.2,0.2",
                    help="proportions plain,mm,etf,both (normalized)")
    args = ap.parse_args()

    mix = [float(x) for x in args.mix.split(",")]
    assert len(mix) == 4, "--mix needs 4 proportions: plain,mm,etf,both"
    tot = sum(mix)
    kinds_pool = ["plain", "mm", "etf", "both"]
    counts = [round(args.n * m / tot) for m in mix]
    while sum(counts) < args.n:
        counts[0] += 1
    while sum(counts) > args.n:
        counts[counts.index(max(counts))] -= 1
    kinds = [k for k, c in zip(kinds_pool, counts) for _ in range(c)]

    rng = random.Random(args.seed)
    rng.shuffle(kinds)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    install = ROOT / "data" / "portfolios"
    install.mkdir(parents=True, exist_ok=True)

    cases = [make_case(i, args.seed, kind, rng, install)
             for i, kind in enumerate(kinds)]
    out = OUT_DIR / f"generated_{args.seed}.jsonl"
    out.write_text("\n".join(json.dumps(c) for c in cases) + "\n")
    # copies of the data files also live beside the dataset for shipping
    for c in cases:
        for f in c["portfolios"]:
            shutil.copy(install / f, OUT_DIR / f)
    by_kind = {k: sum(1 for c in cases if c["kind"] == k) for k in kinds_pool}
    print(f"wrote {len(cases)} cases -> {out}  (mix: {by_kind})")
    print(f"installed {sum(len(c['portfolios']) for c in cases)} data files -> data/portfolios/")


if __name__ == "__main__":
    main()
