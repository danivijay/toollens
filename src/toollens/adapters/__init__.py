"""Model adapters: anything that, given tools and a request, decides which tool to call."""

from toollens.adapters.base import SYSTEM_PROMPT, ModelAdapter
from toollens.adapters.bm25 import BM25Adapter
from toollens.adapters.llm import AnthropicAdapter, OpenAICompatAdapter, make_adapter
from toollens.adapters.replay import ReplayAdapter

__all__ = [
    "SYSTEM_PROMPT",
    "AnthropicAdapter",
    "BM25Adapter",
    "ModelAdapter",
    "OpenAICompatAdapter",
    "ReplayAdapter",
    "make_adapter",
]
