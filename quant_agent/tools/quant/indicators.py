"""Quantitative layer: indicator formulas plus the tool wrapper.

All formulas live here in Python (no external context files). The tool
fetches prices through the shared as-of-truncated loader, so the PIT
guardrail applies to model inputs automatically.
"""
from datetime import date

import numpy as np
import pandas as pd

from ...schemas import Evidence
from ..base import Tool
from ..prices import load_prices


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(int(window)).mean()


def ema(close: pd.Series, window: int) -> pd.Series:
    return close.ewm(span=int(window), adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI with a divide-by-zero guard.

    When a window has no down-days, average loss is 0 and the naive
    100 - 100/(1 + up/down) divides by zero. We compute RS safely and define
    RSI = 100 where there are gains and no losses, RSI = 0 where there are
    losses and no gains, and the neutral 50 where a window is perfectly flat.
    """
    window = int(window)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing (EMA with alpha = 1/window) is the standard RSI average.
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    out = 100 - 100 / (1 + avg_gain / avg_loss)
    # Resolve the cases where avg_loss == 0 (RS is inf/NaN) explicitly.
    out = out.where(avg_loss != 0, other=100.0)            # gains, no losses -> 100
    out = out.where(~((avg_loss == 0) & (avg_gain == 0)), other=50.0)  # flat -> 50
    return out


def realized_vol(close: pd.Series, window: int = 21) -> pd.Series:
    return close.pct_change().rolling(int(window)).std() * np.sqrt(252)


INDICATORS = {"sma": sma, "ema": ema, "rsi": rsi, "realized_vol": realized_vol}


class QuantIndicatorTool(Tool):
    name = "quant_indicator"
    needs_query = True  # ticker-allowlist gate
    description = (
        "quant_indicator(ticker: str, indicator: str, window: int = 14) - compute one of: sma, ema, rsi, "
        "realized_vol (annualized), max_drawdown, correlation, on daily closes up to the as-of date. "
        "For correlation, pass two tickers like 'AAPL,MSFT'. For max_drawdown and correlation, "
        "window is the lookback in trading days (default 252). Only tickers named in the question may be used."
    )

    def run(self, as_of: date, ticker, indicator, window=14, _query="", **_):
        from ..base import assert_ticker_named
        assert_ticker_named(ticker, _query, self.name)
        ind = str(indicator).lower().strip()
        tick = str(ticker).upper().replace(" ", "")
        if ind == "correlation":
            parts = tick.split(",")
            if len(parts) < 2:
                raise ValueError("correlation needs two tickers, e.g. ticker='AAPL,MSFT'.")
            w = int(window) if int(window) > 14 else 252
            t1, t2 = parts[0], parts[1]
            r1 = load_prices(t1, as_of, w * 2)["Close"].pct_change()
            r2 = load_prices(t2, as_of, w * 2)["Close"].pct_change()
            joined = pd.concat([r1, r2], axis=1, join="inner").dropna().tail(w)
            val = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
            content = (
                f"Daily-return correlation of {t1} and {t2} over the last {len(joined)} overlapping "
                f"sessions ending {as_of}: {val:.3f}."
            )
            sid = f"quant:correlation:{t1}-{t2}:{w}:{as_of}"
        elif ind == "max_drawdown":
            w = int(window) if int(window) > 14 else 252
            close = load_prices(tick, as_of, int(w * 1.6) + 30)["Close"].tail(w)
            dd = float((close / close.cummax() - 1).min())
            content = (
                f"Maximum drawdown of {tick} over the last {len(close)} sessions ending {as_of}: {dd * 100:.2f}%."
            )
            sid = f"quant:max_drawdown:{tick}:{w}:{as_of}"
        elif ind in INDICATORS:
            w = int(window)
            close = load_prices(tick, as_of, max(380, w * 6))["Close"]
            series = INDICATORS[ind](close, w).dropna()
            if series.empty:
                raise RuntimeError(f"Not enough history to compute {ind}({w}) for {tick} as of {as_of}.")
            val = float(series.iloc[-1])
            content = f"{ind.upper()}({w}) for {tick} as of {series.index[-1].date()}: {val:.4f}."
            sid = f"quant:{ind}:{tick}:{w}:{as_of}"
        else:
            raise ValueError(f"Unknown indicator '{indicator}'. Choose from: sma, ema, rsi, realized_vol, max_drawdown, correlation.")
        return [
            Evidence(source_id=sid, store="quant", content=content, pub_date=as_of,
                     pit_safe=False, meta={"pit_caveat": "retroactive"})
        ]
