"""Agent-facing equity attribution tool (v11.5, ported from NLPHFEA).

Runs performance attribution over portfolio files in data/portfolios/. Like the
docs corpus, portfolio files carry their publication date in the filename
(2025-03-31_q1_portfolio.csv); a file dated after the as-of date is INVISIBLE -
attribution over a future portfolio snapshot is lookahead, the exact thing this
system exists to prevent. Undated files fall back to file mtime (marked
unreliable in provenance, mirroring doc_kb).

Ops:
  list             - visible portfolio files at the as-of date
  profile          - data shape, long/short detection, suggested method
  brinson_fachler  - long-only relative attribution vs benchmark weights
  cooper_li        - long/short absolute P&L attribution (needs notional)

No network, no tickers fetched: purely file-based computation, so the ticker
allowlist gate does not apply (needs_query=False). Deterministic given the file.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

from ...schemas import Evidence
from ..base import Tool
from . import attribution as attr

_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _portfolios_dir() -> Path:
    from ... import config
    d = config.DATA_DIR / "portfolios"
    d.mkdir(parents=True, exist_ok=True)
    note = d / "put-portfolio-files-here.txt"
    if not note.exists():
        note.write_text(
            "Portfolio holdings files (CSV/XLSX) go here.\n"
            "Prefix filenames with the snapshot date (2025-03-31_q1_portfolio.csv);\n"
            "files dated after the as-of date are invisible to the agent.\n"
            "Expected columns: a grouping column (Sector/Industry/...), Portfolio\n"
            "Weight, a benchmark/universe weight (Cap Weight), and Return.\n"
            "This placeholder is never read.\n")
    return d


def _pub_date(path: Path) -> tuple[date, bool]:
    m = _DATE_RE.search(path.name)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3))), True
    return datetime.fromtimestamp(path.stat().st_mtime).date(), False


def _resolve_file(name: str, as_of: date) -> tuple[Path, date, bool]:
    """Resolve a file name inside data/portfolios/ and enforce the PIT rule.

    Path traversal is rejected (name only, no separators); a post-as-of file is
    refused with the same 'not visible' framing doc_kb uses.
    """
    if "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"portfolio file must be a bare filename in data/portfolios/, got {name!r}")
    d = _portfolios_dir()
    path = d / name
    if not path.exists():
        visible = [p.name for p, pd_, _ in _visible_files(as_of)]
        raise FileNotFoundError(
            f"No portfolio file named {name!r}. Visible files at {as_of}: "
            f"{visible or 'none - add files to data/portfolios/'}")
    pub, reliable = _pub_date(path)
    if pub > as_of:
        raise RuntimeError(
            f"portfolio file {name!r} is dated {pub}, after the as-of date {as_of} - "
            "not visible (attribution over a future snapshot is lookahead).")
    return path, pub, reliable


def _companions(path: Path, as_of: date):
    """Discover companion files sharing the holdings file's basename (v11.7).

    {base}.csv + {base}_trades.csv / {base}_etfs.csv / {base}_etf_constituents.csv.
    Same basename => same filename date => same PIT gate; nothing new to enforce.
    Returns (trades_df|None, etfs_df|None, constituents_df|None).
    """
    stem = path.name[: -len(path.suffix)]
    d = path.parent
    out = {}
    for kind in ("trades", "etfs", "etf_constituents"):
        p = d / f"{stem}_{kind}{path.suffix}"
        out[kind] = attr.load_table(p) if p.exists() else None
    if (out["etfs"] is not None) != (out["etf_constituents"] is not None):
        raise ValueError(
            f"{stem}: _etfs and _etf_constituents companion files must both be "
            "present (ETF decomposition needs the basket).")
    return out["trades"], out["etfs"], out["etf_constituents"]


def _visible_files(as_of: date) -> list[tuple[Path, date, bool]]:
    out = []
    for p in sorted(_portfolios_dir().iterdir()):
        if not p.is_file() or p.name.startswith("put-") or p.name.startswith("."):
            continue
        if p.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            continue
        pub, reliable = _pub_date(p)
        if pub <= as_of:
            out.append((p, pub, reliable))
    return out


def _fmt(x: float, nd: int = 6) -> str:
    return f"{x:.{nd}f}".rstrip("0").rstrip(".")


class AttributionTool(Tool):
    name = "equity_attribution"
    needs_query = False
    description = (
        "equity_attribution(op: str, file: str = None, group_by: str = None, "
        "notional: float = None) - performance attribution over portfolio files in "
        "data/portfolios/ (PIT: files dated after as-of are invisible). op='list' shows "
        "visible files; op='profile' inspects a file and suggests a method; "
        "op='brinson_fachler' decomposes RELATIVE return vs benchmark into "
        "allocation/selection/interaction per group (long-only); op='cooper_li' decomposes "
        "ABSOLUTE long/short P&L into market(NE)/bucket(BE)/selection(LSE,SSE) effects "
        "(pass notional for dollar P&L; long/short portfolios). group_by defaults to "
        "'Sector' when present. Use for 'why did the portfolio out/under-perform' questions."
    )

    def run(self, as_of: date, op, file=None, group_by=None, notional=None, **_):
        op = str(op).lower().strip()

        if op == "list":
            files = _visible_files(as_of)
            hidden = sum(1 for p in _portfolios_dir().iterdir()
                         if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls"}
                         and _pub_date(p)[0] > as_of)
            if not files:
                body = (f"No portfolio files visible at {as_of}"
                        + (f" ({hidden} file(s) hidden: dated after as-of)." if hidden else
                           ". Add files to data/portfolios/."))
            else:
                lines = [f"- {p.name} (dated {pub}{'' if rel else ', date from mtime - unreliable'})"
                         for p, pub, rel in files]
                body = (f"Portfolio files visible at {as_of}:\n" + "\n".join(lines)
                        + (f"\n({hidden} more hidden: dated after as-of)" if hidden else ""))
            return [Evidence(source_id="attribution:list", store="quant", content=body,
                             pub_date=as_of, pit_safe=True, meta={})]

        if not file:
            raise ValueError(f"op={op!r} requires file (use op='list' to see visible files)")
        path, pub, reliable = _resolve_file(str(file), as_of)
        df = attr.load_table(path)

        if op == "profile":
            p = attr.profile(df)
            body = (f"Profile of {path.name} (dated {pub}): {p['rows']} rows; columns "
                    f"{p['columns']}; grouping columns {p['grouping_columns']}; "
                    f"portfolio weights '{p['portfolio_weight_col']}', benchmark/universe "
                    f"weights '{p['benchmark_weight_col']}', returns '{p['return_col']}'.")
            if p.get("portfolio_weight_col"):
                body += (f" Shorts: {p['has_short_positions']}; gross {_fmt(p['gross_exposure'],4)}, "
                         f"net {_fmt(p['net_exposure'],4)}. Suggested method: {p['suggested_method']}"
                         " (Brinson-Fachler is long-only relative; Cooper-Li handles long/short P&L).")
            trades, etfs, cons = _companions(path, as_of)
            comp = [k for k, v in (("trades", trades), ("etfs", etfs)) if v is not None]
            if comp:
                body += (f" Companion files present: {comp} - op='cooper_li' will compute the "
                         "FULL Cooper-Li PDR (base + market-making + ETF arbitrage).")
            return [Evidence(source_id=f"attribution:profile:{path.name}", store="quant",
                             content=body, pub_date=pub, pit_safe=reliable,
                             meta={"pit_caveat": None if reliable else "date_from_mtime"})]

        # Compute ops need resolved columns.
        pw = attr.resolve_col(df, attr.PORT_W_CANDIDATES)
        bw = attr.resolve_col(df, attr.BENCH_W_CANDIDATES)
        rc = attr.resolve_col(df, attr.RETURN_CANDIDATES)
        if not (pw and bw and rc):
            raise ValueError(
                f"could not resolve required columns in {path.name} "
                f"(portfolio_weight={pw}, benchmark_weight={bw}, return={rc}); "
                "run op='profile' to inspect the file.")
        grp_col = group_by or ("Sector" if "Sector" in df.columns else None)
        if not grp_col or grp_col not in df.columns:
            raise ValueError(f"group_by column {group_by!r} not in {path.name}; "
                             f"available grouping columns: "
                             f"{[c for c in df.columns if c in attr.GROUP_CANDIDATES]}")

        if op == "brinson_fachler":
            grp, s = attr.brinson_fachler(df, grp_col, pw, bw, rc)
            rows = "; ".join(
                f"{r.bucket}: alloc={_fmt(r.Allocation)}, sel={_fmt(r.Selection)}, "
                f"inter={_fmt(r.Interaction)}, total={_fmt(r.Total)}"
                for r in grp.itertuples())
            body = (f"Brinson-Fachler attribution of {path.name} (dated {pub}) by {grp_col}: "
                    f"portfolio return {_fmt(s['portfolio_return'])}, benchmark "
                    f"{_fmt(s['benchmark_return'])}, ACTIVE {_fmt(s['active_return'])} = "
                    f"allocation {_fmt(s['total_allocation'])} + selection "
                    f"{_fmt(s['total_selection'])} + interaction {_fmt(s['total_interaction'])} "
                    f"(residual {s['reconciliation_residual']:.1e}). Per {grp_col}: {rows}.")
            sid = f"attribution:bf:{path.name}:{grp_col}"
        elif op == "cooper_li":
            n = float(notional) if notional else 1.0
            trades, etfs, cons = _companions(path, as_of)
            grp, s = attr.cooper_li_full(df, grp_col, pw, bw, rc, notional=n,
                                         trades=trades, etfs=etfs, constituents=cons)
            full = trades is not None or etfs is not None
            grp = grp.rename(columns={grp.columns[0]: "bucket"})
            unit = "USD" if notional else "return units (notional=1)"
            extra = ""
            if full:
                extra = (f" Plus market-making MM {_fmt(s['MM'], 2 if notional else 6)} and "
                         f"ETF arbitrage AE {_fmt(s['AE'], 2 if notional else 6)}: "
                         f"FULL PDR {_fmt(s['PDR'], 2 if notional else 6)} "
                         f"(= base {_fmt(s['base_pnl'], 2 if notional else 6)} + MM + AE; "
                         f"Cooper-Li eqn 11).")
            rows = "; ".join(
                f"{r.bucket}: NE={_fmt(r.NE,2 if notional else 6)}, BE={_fmt(r.BE,2 if notional else 6)}, "
                f"LSE={_fmt(r.LSE,2 if notional else 6)}, SSE={_fmt(r.SSE,2 if notional else 6)}, "
                f"pnl={_fmt(r.bucket_pnl,2 if notional else 6)}"
                for r in grp.itertuples())
            nd = 2 if notional else 6
            body = (f"Cooper-Li attribution of {path.name} (dated {pub}) by {grp_col}, in {unit}: "
                    f"TOTAL P&L {_fmt(s.get('total_pnl', s.get('base_pnl'))    ,nd)} = market NE {_fmt(s['NE_total'],nd)} + "
                    f"bucket BE {_fmt(s['BE_total'],nd)} + long selection LSE {_fmt(s['LSE_total'],nd)} + "
                    f"short selection SSE {_fmt(s['SSE_total'],nd)} "
                    f"(residual {s['reconciliation_residual']:.1e}; gross {_fmt(s['gross_exposure'],4)}, "
                    f"net {_fmt(s['net_exposure'],4)}, universe return {_fmt(s['universe_return'])}). "
                    f"Per {grp_col}: {rows}.{extra}")
            sid = f"attribution:cl:{path.name}:{grp_col}:{_fmt(n,4)}" + (":full" if full else "")
        else:
            raise ValueError("op must be 'list', 'profile', 'brinson_fachler', or 'cooper_li'")

        return [Evidence(source_id=sid, store="quant", content=body, pub_date=pub,
                         pit_safe=reliable,
                         meta={"pit_caveat": None if reliable else "date_from_mtime"})]
