"""Tiny, dependency-free text helpers used by the lint, the BM25 router and the template generator."""

from __future__ import annotations

import math
import re
from collections import Counter

from toollens.models import ToolSpec

STOPWORDS = frozenset(
    """a an the and or of to in on for with by at from as is are be been was were it its this that these
    those i me my we our you your he she they them their can could would should will shall do does did
    please just any some all into about over under up out if then than so not no yes what which who whom
    when where why how there here have has had get gets got via per also use used using one two may might""".split()
)

_SPLIT_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WORD = re.compile(r"[a-z0-9]+")


def split_identifier(name: str) -> list[str]:
    """``searchEvents`` / ``search_events`` / ``search-events`` -> ['search', 'events']."""
    name = _SPLIT_CAMEL.sub(" ", name)
    return [w for w in re.split(r"[^A-Za-z0-9]+", name.lower()) if w]


def humanize(name: str) -> str:
    return " ".join(split_identifier(name))


def stem(word: str) -> str:
    """A deliberately crude suffix stripper. Good enough to make 'meetings' match 'meeting'."""
    for suffix in ("ies", "ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            if suffix == "ies":
                return word[: -len(suffix)] + "y"
            return word[: -len(suffix)]
    return word


def tokens(text: str, *, keep_stopwords: bool = False) -> list[str]:
    text = _SPLIT_CAMEL.sub(" ", text).lower()
    words = _WORD.findall(text)
    if not keep_stopwords:
        words = [w for w in words if w not in STOPWORDS]
    # dates, ids and counts carry no tool semantics; single letters are contraction debris ("what's")
    words = [w for w in words if not w.isdigit() and len(w) > 1]
    return [stem(w) for w in words]


def tool_document(tool: ToolSpec) -> list[str]:
    """Everything a model sees about a tool, flattened to tokens."""
    parts = [humanize(tool.name), tool.description]
    for pname, pschema in tool.properties.items():
        parts.append(humanize(pname))
        if isinstance(pschema, dict):
            parts.append(str(pschema.get("description", "")))
            if "enum" in pschema:
                parts.append(" ".join(map(str, pschema["enum"])))
    return tokens(" ".join(parts))


def cosine(a: Counter[str], b: Counter[str], idf: dict[str, float] | None = None) -> float:
    if not a or not b:
        return 0.0
    w = idf or {}
    dot = sum(a[t] * b[t] * w.get(t, 1.0) ** 2 for t in a.keys() & b.keys())
    na = math.sqrt(sum((v * w.get(t, 1.0)) ** 2 for t, v in a.items()))
    nb = math.sqrt(sum((v * w.get(t, 1.0)) ** 2 for t, v in b.items()))
    return dot / (na * nb) if na and nb else 0.0


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token for English/JSON). Labelled '≈' wherever shown."""
    return max(1, math.ceil(len(text) / 4))
