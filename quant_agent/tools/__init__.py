"""Tool registry. The planner sees describe_tools(); the orchestrator
executes by name through REGISTRY. Adding a store = adding one entry here.
"""
from .charts import ChartTool
from .doc_kb import DocSearchTool, KnowledgeSearchTool
from .edgar import EdgarFetchTool
from .prices import PriceHistoryTool
from .quant.custom_model import CustomModelTool
from .quant.indicators import QuantIndicatorTool
from .quant.attribution_tool import AttributionTool
from .quant.option_tool import OptionTool

REGISTRY = {
    t.name: t
    for t in (
        PriceHistoryTool(),
        DocSearchTool(),
        KnowledgeSearchTool(),
        EdgarFetchTool(),
        QuantIndicatorTool(),
        OptionTool(),
        AttributionTool(),
        ChartTool(),
        CustomModelTool(),
    )
}


def describe_tools() -> str:
    return "\n".join(f"- {t.description}" for t in REGISTRY.values())
