"""Static analysis of a tool surface: things that predict agent confusion before you run a single model."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from enum import Enum
from itertools import combinations

from pydantic import BaseModel

from toollens.models import ServerSnapshot, ToolSpec
from toollens.text import cosine, estimate_tokens, jaccard, split_identifier, tool_document


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class Finding(BaseModel):
    code: str
    severity: Severity
    tools: list[str]
    message: str


class SimilarPair(BaseModel):
    a: str
    b: str
    description_similarity: float
    name_overlap: float
    disambiguated: bool

    @property
    def risk(self) -> float:
        """0..1. High when tools look alike and neither description points at the other."""
        base = max(self.description_similarity, 0.8 * self.name_overlap)
        return round(base * (0.5 if self.disambiguated else 1.0), 3)


class LintReport(BaseModel):
    findings: list[Finding]
    pairs: list[SimilarPair]
    tool_tokens: dict[str, int]

    @property
    def total_tokens(self) -> int:
        return sum(self.tool_tokens.values())

    def count(self, severity: Severity) -> int:
        return sum(1 for f in self.findings if f.severity == severity)

    def risky_pairs(self, threshold: float = 0.35) -> list[SimilarPair]:
        return [p for p in self.pairs if p.risk >= threshold]


GENERIC_NAMES = {
    "run",
    "execute",
    "query",
    "do",
    "call",
    "process",
    "handle",
    "action",
    "tool",
    "invoke",
    "get",
}
GUIDANCE = re.compile(
    r"\b(use (this|it) (when|to|for|if)|use when|use for|use instead|instead of|not for|do not use|don't use|"
    r"only (when|for|if)|prefer)\b",
    re.IGNORECASE,
)
MIN_DESCRIPTION_CHARS = 40
LARGE_TOOL_TOKENS = 400


def tool_definition_text(tool: ToolSpec) -> str:
    """Roughly what a model is shown for one tool."""
    return json.dumps(
        {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _mentions(text: str, other: ToolSpec) -> bool:
    """Does `text` refer to `other` by name? Whole words only, so 'research' never counts as 'search'."""

    def whole(needle: str, haystack: str) -> bool:
        return bool(needle) and re.search(rf"(?<![\w-]){re.escape(needle)}(?![\w-])", haystack) is not None

    return whole(other.name, text) or whole(" ".join(split_identifier(other.name)), text.lower())


def similar_pairs(tools: list[ToolSpec]) -> list[SimilarPair]:
    docs = {t.name: Counter(tool_document(t)) for t in tools}
    n = len(tools)
    df: Counter[str] = Counter()
    for c in docs.values():
        df.update(c.keys())
    idf = {term: math.log(1 + n / (1 + freq)) + 0.1 for term, freq in df.items()}

    pairs: list[SimilarPair] = []
    for a, b in combinations(tools, 2):
        desc_sim = cosine(docs[a.name], docs[b.name], idf)
        name_sim = jaccard(set(split_identifier(a.name)), set(split_identifier(b.name)))
        disamb = _mentions(a.description, b) or _mentions(b.description, a)
        pairs.append(
            SimilarPair(
                a=a.name,
                b=b.name,
                description_similarity=round(desc_sim, 3),
                name_overlap=round(name_sim, 3),
                disambiguated=disamb,
            )
        )
    pairs.sort(key=lambda p: p.risk, reverse=True)
    return pairs


def lint(snapshot: ServerSnapshot, pair_threshold: float = 0.35) -> LintReport:
    findings: list[Finding] = []
    tools = snapshot.tools

    names = [t.name for t in tools]
    for dup in sorted({n for n in names if names.count(n) > 1}):
        findings.append(
            Finding(code="TL000", severity=Severity.ERROR, tools=[dup], message="duplicate tool name")
        )

    for t in tools:
        desc = t.description.strip()
        if not desc:
            findings.append(
                Finding(
                    code="TL001", severity=Severity.ERROR, tools=[t.name], message="tool has no description"
                )
            )
        elif len(desc) < MIN_DESCRIPTION_CHARS:
            findings.append(
                Finding(
                    code="TL001",
                    severity=Severity.WARNING,
                    tools=[t.name],
                    message=f"description is only {len(desc)} characters; say what it does and when to use it",
                )
            )

        undocumented = [
            p for p, s in t.properties.items() if not (isinstance(s, dict) and s.get("description"))
        ]
        if undocumented:
            findings.append(
                Finding(
                    code="TL002",
                    severity=Severity.WARNING if len(undocumented) == len(t.properties) else Severity.INFO,
                    tools=[t.name],
                    message="parameters without descriptions: " + ", ".join(undocumented),
                )
            )

        missing_required = [r for r in t.required if r not in t.properties]
        if missing_required:
            findings.append(
                Finding(
                    code="TL006",
                    severity=Severity.ERROR,
                    tools=[t.name],
                    message="required parameters not defined in properties: " + ", ".join(missing_required),
                )
            )

        if split_identifier(t.name) and all(w in GENERIC_NAMES for w in split_identifier(t.name)):
            findings.append(
                Finding(
                    code="TL007",
                    severity=Severity.INFO,
                    tools=[t.name],
                    message="generic tool name; models lean on names more than you think",
                )
            )

    pairs = similar_pairs(tools)
    for p in pairs:
        if p.risk < pair_threshold:
            continue
        hint = (
            "descriptions reference each other, which helps"
            if p.disambiguated
            else f"neither description says when to pick one over the other (e.g. 'use {p.b} instead when ...')"
        )
        findings.append(
            Finding(
                code="TL003",
                severity=Severity.WARNING if not p.disambiguated else Severity.INFO,
                tools=[p.a, p.b],
                message=(
                    f"look alike (description similarity {p.description_similarity:.2f}, "
                    f"name overlap {p.name_overlap:.2f}); {hint}"
                ),
            )
        )

    risky_tools = {name for p in pairs if p.risk >= pair_threshold for name in (p.a, p.b)}
    for t in tools:
        if t.name in risky_tools and t.description and not GUIDANCE.search(t.description):
            findings.append(
                Finding(
                    code="TL004",
                    severity=Severity.INFO,
                    tools=[t.name],
                    message="has look-alike siblings but no 'use this when / not for' guidance",
                )
            )

    tool_tokens = {t.name: estimate_tokens(tool_definition_text(t)) for t in tools}
    for name, tok in tool_tokens.items():
        if tok > LARGE_TOOL_TOKENS:
            findings.append(
                Finding(
                    code="TL005",
                    severity=Severity.INFO,
                    tools=[name],
                    message=f"definition is ≈{tok} tokens; large schemas crowd small models' context",
                )
            )

    order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
    findings.sort(key=lambda f: (order[f.severity], f.code, f.tools))
    return LintReport(findings=findings, pairs=pairs, tool_tokens=tool_tokens)
