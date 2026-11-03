"""Tool wrapper for the proprietary model in model_core/.

Guarantees regardless of what the model does: inputs truncated at as-of,
output emitted as versioned Evidence (the version in the source id is what
keeps old results attributable to the code that produced them).
"""
from ...schemas import Evidence
from ..base import Tool
from ..prices import load_prices
from .model_core import model


class CustomModelTool(Tool):
    name = "run_model"
    needs_query = True  # ticker-allowlist gate
    description = (
        f"run_model(ticker: str, params: dict = {{}}) - run the proprietary model "
        f"({model.MODEL_NAME} v{model.MODEL_VERSION}) on point-in-time prices and report its output. "
        "Never reimplement or approximate the model; always call this tool. "
        "params are passed straight to the model (defaults live in the model). "
        "Only tickers named in the user's question may be used."
    )

    def run(self, as_of, ticker, params=None, _query="", **_):
        from ..base import assert_ticker_named
        assert_ticker_named(ticker, _query, self.name)
        tick = str(ticker).upper()
        params = dict(params or {})
        df = load_prices(tick, as_of, 730)
        result = model.predict(df, **params)
        if not isinstance(result, dict):
            raise RuntimeError(f"{model.MODEL_NAME} predict() must return a flat dict, got {type(result).__name__}.")
        rendered = ", ".join(f"{k}={v}" for k, v in result.items())
        param_str = ",".join(f"{k}={v}" for k, v in sorted(params.items())) or "defaults"
        return [
            Evidence(
                source_id=f"model:{model.MODEL_NAME}:v{model.MODEL_VERSION}:{tick}:{param_str}:{as_of}",
                store="model",
                content=(f"{model.MODEL_NAME} v{model.MODEL_VERSION} for {tick} as of {as_of} "
                         f"({len(df)} sessions, params: {param_str}): {rendered}."),
                pub_date=as_of,
                pit_safe=False,  # model inputs are retroactively-adjusted prices
                kind="fact",
                meta={"model": model.MODEL_NAME, "version": model.MODEL_VERSION,
                      "params": params, "pit_caveat": "retroactive"},
            )
        ]
