"""Turn predictions into numbers: accuracy, per-tool precision/recall, confusion matrix, confused pairs."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

import jsonschema
from pydantic import BaseModel, Field

from toollens.models import ERROR, NO_TOOL, Prediction, ServerSnapshot, Task, TaskKind, TaskResult

ANY = "*"
"""In expected_args, a value of "*" means 'this argument must be present, any value'."""


def _norm(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.lower().split())
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in value.items()}
    return value


_IDENTIFIER = re.compile(r"^[^\s]*[\d_@./:-][^\s]*$")


def _value_matches(expected: Any, actual: Any) -> bool:
    if expected == ANY:
        return True
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        # every expected item must appear somewhere in the actual list (order and extras don't matter)
        return all(any(_value_matches(e, a) for a in actual) for e in expected)
    if isinstance(expected, dict):
        return isinstance(actual, dict) and args_contain(expected, actual)
    e, a = _norm(expected), _norm(actual)
    if isinstance(e, str) and isinstance(a, str):
        if e == a:
            return True
        # Free text may be embedded in a longer value ("budget" in "Q3 budget review"), but identifiers,
        # emails, dates and paths must match exactly so that "evt_1" never matches "evt_1041".
        return bool(e) and not _IDENTIFIER.match(e) and e in a
    return a == e


def args_contain(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """True when every expected key is present in actual with a matching value.

    * Strings match case-insensitively; free-text values also match as substrings, so
      ``{"query": "budget"}`` matches ``{"query": "Q3 budget review"}``. Identifier-like values
      (containing digits, ``_``, ``@``, ``.``, ``/``, ``:`` or ``-``) must match exactly.
    * Numbers compare by value (``3 == 3.0``).
    * Lists must contain every expected item, in any order; extra items are allowed.
    * ``"*"`` means "present, any value".
    """
    return all(key in actual and _value_matches(exp, actual[key]) for key, exp in expected.items())


def score_one(snapshot: ServerSnapshot, task: Task, pred: Prediction, produces_arguments: bool) -> TaskResult:
    tool_correct = pred.error is None and pred.tool == task.expected_tool
    args_valid: bool | None = None
    args_match: bool | None = None
    if produces_arguments and tool_correct and task.expected_tool is not None:
        spec = snapshot.tool(task.expected_tool)
        if spec is not None:
            try:
                jsonschema.validate(pred.arguments, spec.input_schema or {"type": "object"})
                args_valid = True
            except jsonschema.ValidationError:
                args_valid = False
            except jsonschema.SchemaError:
                args_valid = None
        args_match = args_contain(task.expected_args, pred.arguments)
    return TaskResult(
        task=task, prediction=pred, tool_correct=tool_correct, args_valid=args_valid, args_match=args_match
    )


def score(
    snapshot: ServerSnapshot,
    tasks: list[Task],
    predictions: list[Prediction],
    *,
    produces_arguments: bool = True,
) -> list[TaskResult]:
    by_id = {p.task_id: p for p in predictions}
    missing = [t.id for t in tasks if t.id not in by_id]
    if missing:
        raise ValueError(f"no prediction for tasks: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    return [score_one(snapshot, t, by_id[t.id], produces_arguments) for t in tasks]


class ToolStats(BaseModel):
    tool: str
    support: int
    predicted: int
    correct: int

    @property
    def precision(self) -> float | None:
        return self.correct / self.predicted if self.predicted else None

    @property
    def recall(self) -> float | None:
        return self.correct / self.support if self.support else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if not p or not r:
            return 0.0 if (p is not None and r is not None) else None
        return 2 * p * r / (p + r)


class ConfusedPair(BaseModel):
    a: str
    b: str
    a_as_b: int
    """Tasks that needed `a` but the model called `b`."""
    b_as_a: int
    examples: list[str] = Field(default_factory=list)
    """Task ids, for drilling in."""

    @property
    def total(self) -> int:
        return self.a_as_b + self.b_as_a


class Metrics(BaseModel):
    model: str
    n: int
    accuracy: float
    accuracy_by_kind: dict[str, float]
    count_by_kind: dict[str, int]
    abstention_precision: float | None
    """Of the times the model called no tool, how often was that right?"""
    abstention_recall: float | None
    """Of the no-tool tasks, how often did the model correctly call nothing?"""
    args_valid_rate: float | None
    args_match_rate: float | None
    missed_calls: int = 0
    """Tasks that needed a tool where the model called nothing."""
    false_calls: int = 0
    """Out-of-scope tasks where the model called a tool anyway."""
    errors: int
    mean_latency_ms: float
    labels: list[str]
    confusion: dict[str, dict[str, int]]
    per_tool: list[ToolStats]
    confused_pairs: list[ConfusedPair]


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(model: str, snapshot: ServerSnapshot, results: list[TaskResult]) -> Metrics:
    n = len(results)
    by_kind: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_kind[r.task.kind.value].append(r.tool_correct)

    abstained = [r for r in results if r.prediction.error is None and r.prediction.tool is None]
    no_tool_tasks = [r for r in results if r.task.kind == TaskKind.NO_TOOL or r.task.expected_tool is None]

    labels = list(snapshot.tool_names) + [NO_TOOL]
    if any(r.prediction.error for r in results):
        labels.append(ERROR)
    confusion: dict[str, dict[str, int]] = {e: {p: 0 for p in labels} for e in labels if e != ERROR}
    for r in results:
        e, p = r.expected_label, r.predicted_label
        confusion.setdefault(e, {lbl: 0 for lbl in labels})
        confusion[e][p] = confusion[e].get(p, 0) + 1

    support = Counter(r.expected_label for r in results)
    predicted = Counter(r.predicted_label for r in results)
    correct = Counter(r.expected_label for r in results if r.tool_correct)
    per_tool = [
        ToolStats(tool=t, support=support.get(t, 0), predicted=predicted.get(t, 0), correct=correct.get(t, 0))
        for t in snapshot.tool_names
    ]

    pair_counts: dict[tuple[str, str], list[TaskResult]] = defaultdict(list)
    for r in results:
        e, p = r.expected_label, r.predicted_label
        if e != p and ERROR not in (e, p) and NO_TOOL not in (e, p):
            pair_counts[(e, p)].append(r)
    seen: set[frozenset[str]] = set()
    pairs: list[ConfusedPair] = []
    for a, b in pair_counts:
        key = frozenset((a, b))
        if key in seen:
            continue
        seen.add(key)
        ab, ba = pair_counts.get((a, b), []), pair_counts.get((b, a), [])
        pairs.append(
            ConfusedPair(
                a=a, b=b, a_as_b=len(ab), b_as_a=len(ba), examples=[r.task.id for r in (ab + ba)][:6]
            )
        )
    pairs.sort(key=lambda p: (-p.total, p.a, p.b))

    return Metrics(
        model=model,
        n=n,
        accuracy=sum(r.tool_correct for r in results) / n if n else 0.0,
        accuracy_by_kind={k: sum(v) / len(v) for k, v in sorted(by_kind.items())},
        count_by_kind={k: len(v) for k, v in sorted(by_kind.items())},
        abstention_precision=_rate([r.task.expected_tool is None for r in abstained]),
        abstention_recall=_rate(
            [r.prediction.tool is None and r.prediction.error is None for r in no_tool_tasks]
        ),
        args_valid_rate=_rate([r.args_valid for r in results if r.args_valid is not None]),
        args_match_rate=_rate([r.args_match for r in results if r.args_match is not None]),
        missed_calls=sum(
            1
            for r in results
            if r.task.expected_tool and r.prediction.tool is None and not r.prediction.error
        ),
        false_calls=sum(
            1
            for r in results
            if r.task.expected_tool is None and r.prediction.tool is not None and not r.prediction.error
        ),
        errors=sum(1 for r in results if r.prediction.error),
        mean_latency_ms=round(sum(r.prediction.latency_ms for r in results) / n, 2) if n else 0.0,
        labels=labels,
        confusion=confusion,
        per_tool=per_tool,
        confused_pairs=pairs,
    )
