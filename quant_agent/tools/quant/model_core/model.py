"""Adapter between the run_model tool and YOUR proprietary code.

Replace the body of predict() with a call into your model (drop your files
into this model_core/ folder). Keep the signature and flat-dict return; the
wrapper handles point-in-time inputs, provenance, and tracing.
"""
import pandas as pd

MODEL_NAME = "my_model"      # TODO: your model's name
MODEL_VERSION = "0.1.0"      # TODO: bump on every model change
DEFAULT_PARAMS = {"window": 21}


def predict(prices: pd.DataFrame, **params) -> dict:
    """Run the model on an ALREADY as-of-truncated daily OHLCV DataFrame.
    Return a flat dict of named outputs, e.g. {"signal": 0.42, "label": "long"}.

    TODO: replace the placeholder below, e.g.:
        from .my_strategy import MyStrategy
        return MyStrategy(**params).run(prices)
    """
    p = {**DEFAULT_PARAMS, **params}
    w = int(p["window"])
    # --- placeholder model: trailing-return z-score (replace me) ---
    rets = prices["Close"].pct_change().dropna()
    if len(rets) < w * 2:
        raise ValueError(f"Need at least {w * 2} sessions of history, got {len(rets)}.")
    z = float((rets.tail(w).mean() - rets.mean()) / (rets.std() / w ** 0.5))
    return {"zscore": round(z, 4), "window": w, "sessions_used": len(rets)}
