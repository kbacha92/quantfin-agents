"""Market data store backed by yfinance. Data is truncated at the as-of date.
"""
from datetime import date, timedelta

from ..schemas import Evidence
from .base import Tool


def _cache_path(ticker: str, as_of: date, lookback_days: int):
    from .. import config

    return config.API_CACHE_DIR / f"yfinance_{ticker.upper()}_{as_of}_{int(lookback_days)}d.csv"


def load_prices(ticker: str, as_of: date, lookback_days: int = 365):
    """Live pull with a point-in-time snapshot.

    The as-of-truncated frame is written to data/api_cache/ on first fetch, and
    identical requests replay the snapshot instead of re-hitting the API. That
    makes reruns reproducible (retroactive vendor adjustments are frozen at
    first pull) and the cache file is the audit record of exactly what the
    system saw.

    Today's snapshot is treated as provisional (fix #8): when as_of is the
    current date, the session is still in progress / EOD data may post after a
    morning pull, so a re-request re-fetches and overwrites rather than serving
    a frozen morning snapshot. Snapshots for *past* as-of dates are immutable -
    that history doesn't change, so they always replay. Delete any snapshot file
    to force a fresh pull regardless.

    OFFLINE-STRICT MODE (Option A, v4): if QUANT_OFFLINE_ONLY is set (the eval
    harness sets it), this function NEVER hits the network. It replays an
    existing snapshot and raises loudly if none exists. This guarantees a
    point-in-time evaluation can never silently measure live data - e.g. if the
    agent mutates a ticker (SYN -> SYNB) to one with no snapshot, the call fails
    instead of quietly fetching real market data.
    """
    import os
    import pandas as pd

    cache = _cache_path(ticker, as_of, lookback_days)
    offline_only = os.environ.get("QUANT_OFFLINE_ONLY", "").strip().lower() in ("1", "true", "yes")
    is_today = as_of >= date.today()

    if cache.exists() and (offline_only or not is_today):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if not df.empty:
            return df

    if offline_only:
        # Fail closed: no snapshot, and we are forbidden from going live.
        raise RuntimeError(
            f"offline-strict mode: no point-in-time snapshot for {ticker} "
            f"({as_of}, {int(lookback_days)}d) and live fetch is disabled. "
            "If the ticker is unexpected, the agent may have altered it; if it is "
            "correct, generate the snapshot first."
        )

    import yfinance as yf

    start = as_of - timedelta(days=int(lookback_days))
    end = as_of + timedelta(days=1)  # yfinance end is exclusive
    df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False, auto_adjust=True)
    if df is None or df.empty:
        # If the live pull fails but we have a prior (today) snapshot, fall back
        # to it rather than erroring - stale-but-present beats nothing.
        if cache.exists():
            prev = pd.read_csv(cache, index_col=0, parse_dates=True)
            if not prev.empty:
                return prev
        raise RuntimeError(f"No price data for {ticker} on or before {as_of}.")
    if hasattr(df.columns, "levels"):
        df.columns = [c[0] for c in df.columns]
    df = df[[d <= as_of for d in df.index.date]]
    if df.empty:
        raise RuntimeError(f"No price data for {ticker} on or before {as_of}.")
    df.to_csv(cache)
    return df


class PriceHistoryTool(Tool):
    name = "price_history"
    needs_query = True  # receives the user's query for the ticker-allowlist gate
    description = (
        "price_history(ticker: str, lookback_days: int = 90) - daily adjusted OHLCV via yfinance, "
        "truncated at the as-of date. Returns summary statistics and the last 5 rows. "
        "Only tickers explicitly named in the user's question may be used."
    )

    def run(self, as_of, ticker, lookback_days=90, _query="", **_):
        from .base import assert_ticker_named
        assert_ticker_named(ticker, _query, self.name)
        df = load_prices(ticker, as_of, lookback_days)
        close = df["Close"]
        rets = close.pct_change().dropna()
        summary = (
            f"{ticker} daily prices, {df.index[0].date()} to {df.index[-1].date()} "
            f"({len(df)} sessions, auto-adjusted; adjustments are retroactive, not strictly PIT).\n"
            f"Last close: {float(close.iloc[-1]):.2f}. "
            f"Period return: {(float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100:.2f}%. "
            f"Annualized vol: {float(rets.std()) * 252 ** 0.5 * 100:.2f}%.\n"
            f"Last 5 rows:\n{df.tail().round(2).to_string()}"
        )
        return [
            Evidence(
                source_id=f"yfinance:{ticker.upper()}:{as_of}:{int(lookback_days)}d",
                store="prices",
                content=summary,
                pub_date=as_of,
                pit_safe=False,  # adjusted closes use retroactive corporate-action adjustments
                meta={
                    "rows": len(df),
                    "snapshot": str(_cache_path(ticker, as_of, lookback_days)),
                    "pit_caveat": "retroactive",
                },
            )
        ]
