"""Chart tool: renders point-in-time figures to OUTPUTS_DIR as PNGs.

Charts are derived artifacts (precedence below data): the Evidence block
records the saved path so the answer can reference it and the UI can
display it.
"""
from .. import config
from ..schemas import Evidence
from .base import Tool
from .prices import load_prices


class ChartTool(Tool):
    name = "make_chart"
    needs_query = True  # ticker-allowlist gate
    description = (
        "make_chart(ticker: str, chart: str = 'price', window: int = 14, lookback_days: int = 252) - "
        "render a PNG saved to the outputs folder and return its path. chart is one of: "
        "'price' (close line), 'drawdown' (underwater plot), 'sma' (close + moving average), "
        "'rsi' (RSI panel). Use when the user asks for a plot, chart, graph, or figure."
    )

    def run(self, as_of, ticker, chart="price", window=14, lookback_days=252, _query="", **_):
        from .base import assert_ticker_named
        assert_ticker_named(ticker, _query, self.name)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tick = str(ticker).upper()
        chart = str(chart).lower()
        w = int(window)
        df = load_prices(tick, as_of, int(lookback_days) + 30)
        close = df["Close"].tail(int(lookback_days))

        fig, ax = plt.subplots(figsize=(9, 4.5))
        if chart == "price":
            ax.plot(close.index, close.values)
            ax.set_title(f"{tick} close, {close.index[0].date()} to {close.index[-1].date()} (as of {as_of})")
            desc = "close-price line"
        elif chart == "drawdown":
            dd = close / close.cummax() - 1
            ax.fill_between(dd.index, dd.values, 0, alpha=0.5)
            ax.set_title(f"{tick} drawdown (as of {as_of})")
            desc = f"underwater plot, max drawdown {float(dd.min()) * 100:.2f}%"
        elif chart == "sma":
            ax.plot(close.index, close.values, label="close")
            ax.plot(close.index, close.rolling(w).mean().values, label=f"SMA({w})")
            ax.legend()
            ax.set_title(f"{tick} close vs SMA({w}) (as of {as_of})")
            desc = f"close with SMA({w}) overlay"
        elif chart == "rsi":
            from .quant.indicators import rsi as _rsi

            rsi = _rsi(close, w).dropna()
            if rsi.empty:
                raise RuntimeError(f"Not enough history to compute RSI({w}) for {tick} as of {as_of}.")
            ax.plot(rsi.index, rsi.values)
            ax.axhline(70, ls="--", lw=0.8)
            ax.axhline(30, ls="--", lw=0.8)
            ax.set_ylim(0, 100)
            ax.set_title(f"{tick} RSI({w}) (as of {as_of})")
            desc = f"RSI({w}) panel, last value {float(rsi.iloc[-1]):.1f}"
        else:
            raise ValueError(f"Unknown chart '{chart}'. Choose: price, drawdown, sma, rsi.")
        fig.autofmt_xdate()
        fig.tight_layout()
        suffix = f"_{w}" if chart in ("sma", "rsi") else ""
        path = config.OUTPUTS_DIR / f"{as_of}_{tick}_{chart}{suffix}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)

        return [
            Evidence(
                source_id=f"chart:{chart}{suffix}:{tick}:{as_of}",
                store="chart",
                content=f"Saved {tick} {chart} chart ({desc}, data through {close.index[-1].date()}) to {path}.",
                pub_date=as_of,
                pit_safe=False,  # rendered from retroactively-adjusted prices
                kind="fact",
                meta={"path": str(path), "pit_caveat": "retroactive"},
            )
        ]
