"""An offline, deterministic baseline: route each request to the tool with the best BM25 score.

This is exactly how retrieval-based "tool RAG" routers pre-select tools for an agent, so it is
a meaningful floor rather than a toy: if BM25 cannot separate two of your tools, a retriever
in front of an LLM probably cannot either. It needs no GPU, no API key and no network, which
also makes it the adapter the test-suite runs against.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from toollens.adapters.base import ModelAdapter
from toollens.models import ServerSnapshot, Task
from toollens.text import tokens, tool_document


class BM25Index:
    def __init__(self, docs: dict[str, list[str]], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = {name: Counter(toks) for name, toks in docs.items()}
        self.lengths = {name: len(toks) for name, toks in docs.items()}
        self.avgdl = (sum(self.lengths.values()) / len(docs)) if docs else 0.0
        n = len(docs)
        df: Counter[str] = Counter()
        for c in self.docs.values():
            df.update(c.keys())
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, tf in self.docs.items():
            dl = self.lengths[name] or 1
            s = 0.0
            for term in query:
                f = tf.get(term, 0)
                if not f:
                    continue
                s += (
                    self.idf[term]
                    * f
                    * (self.k1 + 1)
                    / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                )
            out[name] = s
        return out


class BM25Adapter(ModelAdapter):
    """Lexical tool router. Abstains (calls no tool) when the best score is below ``min_score``."""

    concurrency = 64
    produces_arguments = False

    def __init__(self, min_score: float = 1.0, label: str | None = None) -> None:
        self.name = label or "bm25"
        self.min_score = min_score
        self._cache: dict[tuple, BM25Index] = {}

    def _index(self, snapshot: ServerSnapshot) -> BM25Index:
        key = tuple((t.name, t.description, repr(t.input_schema)) for t in snapshot.tools)
        if key not in self._cache:
            self._cache[key] = BM25Index({t.name: tool_document(t) for t in snapshot.tools})
        return self._cache[key]

    async def _choose(self, snapshot: ServerSnapshot, task: Task) -> tuple[str | None, dict[str, Any], Any]:
        scores = self._index(snapshot).scores(tokens(task.prompt))
        if not scores:
            return None, {}, {}
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        best, best_score = ranked[0]
        raw = {"top": [[n, round(s, 3)] for n, s in ranked[:3]]}
        if best_score < self.min_score:
            return None, {}, raw
        return best, {}, raw
