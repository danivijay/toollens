"""Rich terminal output."""

from __future__ import annotations

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from toollens.lint import LintReport, Severity
from toollens.models import ERROR, NO_TOOL, ServerSnapshot
from toollens.runner import ModelRun, Run

SEV_STYLE = {Severity.ERROR: "bold red", Severity.WARNING: "yellow", Severity.INFO: "dim"}


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def _acc_style(x: float | None) -> str:
    if x is None:
        return "dim"
    return "bold green" if x >= 0.9 else "bold yellow" if x >= 0.7 else "bold red"


def snapshot_panel(snapshot: ServerSnapshot, lint: LintReport) -> Panel:
    t = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False)
    t.add_column("Tool", style="cyan", no_wrap=True)
    t.add_column("Description", overflow="fold")
    t.add_column("≈ tokens", justify="right")
    for tool in snapshot.tools:
        t.add_row(tool.name, tool.description or "[red](none)[/]", str(lint.tool_tokens.get(tool.name, "")))
    title = (
        f"[bold]{snapshot.server_name or snapshot.server}[/] · {len(snapshot.tools)} tools · "
        f"protocol {snapshot.protocol_version or '?'} · tool list ≈{lint.total_tokens:,} tokens"
    )
    return Panel(t, title=title, title_align="left", border_style="blue")


def lint_table(lint: LintReport, limit: int | None = None) -> Table:
    t = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False, title_justify="left")
    t.add_column("", no_wrap=True)
    t.add_column("Code", no_wrap=True)
    t.add_column("Tools", style="cyan")
    t.add_column("Finding", overflow="fold")
    findings = lint.findings if limit is None else lint.findings[:limit]
    for f in findings:
        t.add_row(Text(f.severity.value, style=SEV_STYLE[f.severity]), f.code, ", ".join(f.tools), f.message)
    if not findings:
        t.add_row("", "", "", "[green]No findings.[/]")
    return t


def scoreboard(runs: list[Run]) -> Table:
    t = Table(box=box.ROUNDED, title="[bold]Scoreboard[/]", title_justify="left")
    multi = len(runs) > 1
    with_args = any(m.produces_arguments for run in runs for m in run.models)
    if multi:
        t.add_column("Server", style="bold", no_wrap=True)
    t.add_column("Model", style="cyan", no_wrap=True)
    cols = ["Accuracy", "Direct", "Near-miss", "Abstain"]
    if with_args:
        cols += ["Args valid", "Args match"]
    cols += ["Errors", "Latency"]
    for col in cols:
        t.add_column(col, justify="right", no_wrap=True, min_width=len(col))
    for run in runs:
        for m in run.models:
            mt = m.metrics
            k = mt.accuracy_by_kind
            row = [
                Text(pct(mt.accuracy), style=_acc_style(mt.accuracy)),
                pct(k.get("direct")),
                pct(k.get("near_miss")),
                pct(mt.abstention_recall),
            ]
            if with_args:
                row += [pct(mt.args_valid_rate), pct(mt.args_match_rate)]
            row += [
                str(mt.errors),
                "<1 ms" if mt.mean_latency_ms < 1 else f"{mt.mean_latency_ms:,.0f} ms",
            ]
            lead = [run.snapshot.server_name or run.snapshot.server] if multi else []
            t.add_row(*lead, m.model, *row)
    return t


def _short(label: str, width: int = 3) -> str:
    return label[:width]


def confusion_table(m: ModelRun) -> Group:
    labels = m.metrics.labels
    rows = [lbl for lbl in labels if lbl != ERROR]
    keys = {
        lbl: (str(i + 1) if lbl not in (NO_TOOL, ERROR) else ("∅" if lbl == NO_TOOL else "⚠"))
        for i, lbl in enumerate(labels)
    }
    t = Table(
        box=box.SIMPLE,
        pad_edge=False,
        show_edge=False,
        title="expected ↓  predicted →",
        title_justify="left",
        title_style="dim",
        padding=(0, 0),
        collapse_padding=True,
    )
    # Compact on purpose: 10 tools + "no tool" fit in an 80-column terminal / CI log.
    t.add_column("", style="cyan", no_wrap=True, max_width=22, overflow="ellipsis")
    for lbl in labels:
        t.add_column(keys[lbl], justify="right", width=4, no_wrap=True)
    vmax = max([v for r in rows for c, v in m.metrics.confusion.get(r, {}).items() if c != r] + [1])
    for r in rows:
        cells = []
        for c in labels:
            v = m.metrics.confusion.get(r, {}).get(c, 0)
            if not v:
                cells.append(Text("·", style="dim"))
            elif r == c:
                cells.append(Text(str(v), style="bold green"))
            else:
                style = "bold white on red" if v >= max(2, vmax * 0.6) else "red"
                cells.append(Text(str(v), style=style))
        t.add_row((r if r == NO_TOOL else f"{keys[r]:>2} {r}") + " ", *cells)
    return Group(t)


def pairs_table(m: ModelRun, limit: int = 5) -> Table:
    tasks = {r.task.id: r for r in m.results}
    t = Table(
        box=box.SIMPLE_HEAD,
        expand=True,
        pad_edge=False,
        title="[bold]Most confused pairs[/]",
        title_justify="left",
    )
    t.add_column("Pair", style="cyan", no_wrap=True)
    t.add_column("Mistakes", justify="right")
    t.add_column("Example", overflow="fold")
    for p in m.metrics.confused_pairs[:limit]:
        r = tasks[p.examples[0]]
        t.add_row(
            f"{p.a} ⇄ {p.b}",
            f"{p.total}",
            f"“{r.task.prompt}” → [red]{r.predicted_label}[/] (expected {r.expected_label})",
        )
    if not m.metrics.confused_pairs:
        t.add_row("[green]none[/]", "0", "")
    mt = m.metrics
    t.caption = (
        f"Also: {mt.missed_calls} missed calls (needed a tool, called none) · "
        f"{mt.false_calls} false calls (out of scope, called a tool anyway) · {mt.errors} errors"
    )
    t.caption_justify = "left"
    return t


def print_run(console: Console, runs: list[Run], *, matrices: bool = True) -> None:
    console.print(scoreboard(runs))
    for run in runs:
        for m in run.models:
            title = f"[bold]{run.snapshot.server_name or 'server'}[/] · [cyan]{m.model}[/] · accuracy " + pct(
                m.metrics.accuracy
            )
            items = []
            if matrices:
                items.append(confusion_table(m))
            items.append(pairs_table(m))
            console.print(Panel(Group(*items), title=title, title_align="left", border_style="blue"))
