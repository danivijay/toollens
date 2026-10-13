"""`toollens` command line interface."""

from __future__ import annotations

import json
import sys

import anyio
import click
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from toollens import __version__
from toollens.adapters import make_adapter
from toollens.dataset import check_tasks
from toollens.discovery import load_or_discover
from toollens.lint import Severity, lint
from toollens.models import read_tasks, write_tasks
from toollens.report.html import write_html
from toollens.report.terminal import lint_table, print_run, snapshot_panel
from toollens.runner import load_run, run_eval, save_run

console = Console(highlight=False)
err = Console(stderr=True, highlight=False)

server_opt = click.option(
    "--server", "-s", help="MCP server: a command to launch over stdio, or an http(s) URL."
)
tools_opt = click.option(
    "--tools",
    "tools_file",
    type=click.Path(exists=True, dir_okay=False),
    help="A saved snapshot (from `inspect --save`).",
)


def _snapshot(server: str | None, tools_file: str | None):
    try:
        snap = anyio.run(load_or_discover, server, tools_file)
    except Exception as exc:
        err.print(f"[red]Could not load tools:[/] {exc}")
        sys.exit(2)
    if not snap.tools:
        err.print("[yellow]The server exposes no tools.[/]")
    return snap


def _tasks(path: str):
    try:
        tasks = read_tasks(path)
    except Exception as exc:
        err.print(f"[red]Could not read tasks:[/] {exc}")
        sys.exit(2)
    if not tasks:
        err.print(f"[red]{path} contains no tasks.[/]")
        sys.exit(2)
    return tasks


def _provider(spec: str) -> str:
    return spec.partition(":")[0]


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="toollens")
def main() -> None:
    """ToolLens: find out which MCP tools your agents confuse, before your users do."""


@main.command()
@server_opt
@tools_opt
@click.option("--save", type=click.Path(dir_okay=False), help="Write the tool snapshot to this JSON file.")
@click.option("--json", "as_json", is_flag=True, help="Print the lint report as JSON.")
@click.option("--strict", is_flag=True, help="Exit non-zero on lint warnings, not just errors.")
def inspect(server, tools_file, save, as_json, strict):
    """List a server's tools and lint them for confusion risks (no model needed)."""
    snap = _snapshot(server, tools_file)
    report = lint(snap)
    if save:
        snap.save(save)
    if as_json:
        click.echo(
            json.dumps({"snapshot": snap.model_dump(), "lint": report.model_dump(mode="json")}, indent=2)
        )
    else:
        console.print(snapshot_panel(snap, report))
        console.print(lint_table(report))
        if save:
            console.print(f"[dim]Snapshot saved to {save}[/]")
    errors = report.count(Severity.ERROR) + (report.count(Severity.WARNING) if strict else 0)
    sys.exit(1 if errors else 0)


@main.command()
@server_opt
@tools_opt
@click.option(
    "--out", "-o", required=True, type=click.Path(dir_okay=False), help="Where to write tasks (JSONL)."
)
@click.option(
    "--generator",
    type=click.Choice(["llm", "template"]),
    default="llm",
    show_default=True,
    help="'llm' writes realistic tasks with a model; 'template' is offline and deterministic (smoke tests).",
)
@click.option(
    "--model",
    "-m",
    "gen_model",
    default="anthropic:claude-sonnet-4-5",
    show_default=True,
    help="Generator model spec, e.g. anthropic:<model>, openai:<model>, ollama:<model>.",
)
@click.option("--base-url", help="Override the endpoint for the generator model.")
@click.option("--api-key", help="API key (defaults to the provider's environment variable).")
@click.option("--per-tool", default=6, show_default=True, help="Direct tasks per tool.")
@click.option("--near-miss", default=4, show_default=True, help="Near-miss tasks per tool (llm only).")
@click.option("--no-tool", default=8, show_default=True, help="Out-of-scope tasks for the whole server.")
@click.option("--seed", default=7, show_default=True, help="Seed for the template generator.")
def generate(
    server, tools_file, out, generator, gen_model, base_url, api_key, per_tool, near_miss, no_tool, seed
):
    """Generate a labelled synthetic task set from a server's tools."""
    snap = _snapshot(server, tools_file)
    if generator == "template":
        from toollens.generate import generate_template_tasks

        tasks = generate_template_tasks(snap, per_tool=per_tool, no_tool=no_tool, seed=seed)
    else:
        from toollens.generate import ChatClient, generate_llm_tasks

        async def go():
            client = ChatClient(gen_model, base_url=base_url, api_key=api_key)
            try:
                with Progress(
                    TextColumn("[cyan]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    console=err,
                    transient=True,
                ) as prog:
                    bar = prog.add_task("generating", total=len(snap.tools))

                    def tick(tool, done, total):
                        prog.update(bar, completed=done, description=f"generated {tool}")

                    return await generate_llm_tasks(
                        snap, client, direct=per_tool, near_miss=near_miss, no_tool=no_tool, progress=tick
                    )
            finally:
                await client.aclose()

        try:
            tasks = anyio.run(go)
        except Exception as exc:
            err.print(f"[red]Generation failed:[/] {exc}")
            sys.exit(2)
    write_tasks(tasks, out)
    kinds: dict[str, int] = {}
    for t in tasks:
        kinds[t.kind.value] = kinds.get(t.kind.value, 0) + 1
    summary = ", ".join(f"{v} {k}" for k, v in sorted(kinds.items()))
    console.print(
        f"Wrote [bold]{len(tasks)}[/] tasks ({summary}) to [cyan]{out}[/]. Review them before trusting scores."
    )


@main.command()
@server_opt
@tools_opt
@click.option("--tasks", "-t", "tasks_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--model",
    "-m",
    "models",
    multiple=True,
    default=["bm25"],
    show_default=True,
    help="Repeatable. bm25 | ollama:<m> | mlx:<m> | lmstudio:<m> | openai:<m> | anthropic:<m> | "
    "openai-compat:<m> (with --base-url) | replay:<predictions.jsonl>",
)
@click.option("--base-url", help="Endpoint override for OpenAI-compatible models.")
@click.option("--api-key", help="API key (defaults to the provider's environment variable).")
@click.option("--concurrency", type=int, help="Parallel requests per model (default: 1 local, 4 remote).")
@click.option("--out", "-o", "out_dir", default="runs", show_default=True, type=click.Path(file_okay=False))
@click.option("--html", "html_path", type=click.Path(dir_okay=False), help="Also write an HTML report here.")
@click.option(
    "--fail-under",
    type=click.FloatRange(0, 1),
    help="Exit 1 if any model's tool accuracy is below this (0-1). For CI.",
)
@click.option("--no-matrix", is_flag=True, help="Skip confusion matrices in terminal output.")
def run(
    server,
    tools_file,
    tasks_path,
    models,
    base_url,
    api_key,
    concurrency,
    out_dir,
    html_path,
    fail_under,
    no_matrix,
):
    """Run a task set against one or more models and score their tool choices."""
    snap = _snapshot(server, tools_file)
    tasks = _tasks(tasks_path)
    problems = check_tasks(snap, tasks)
    if problems:
        err.print(
            f"[yellow]Task set has {len(problems)} problems (run `toollens validate`):[/] {problems[0]}"
        )
    remote = {_provider(m) for m in models if _provider(m) not in ("bm25", "replay")}
    if (api_key or base_url) and len(remote) > 1:
        err.print(
            "[red]--api-key / --base-url would be sent to every provider in this run "
            f"({', '.join(sorted(remote))}).[/] Use each provider's environment variable instead, "
            "or run one provider at a time."
        )
        sys.exit(2)
    try:
        adapters = [
            make_adapter(m, base_url=base_url, api_key=api_key, concurrency=concurrency) for m in models
        ]
    except ValueError as exc:
        err.print(f"[red]{exc}[/]")
        sys.exit(2)

    with Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=err,
        transient=True,
    ) as prog:
        bars = {a.name: prog.add_task(a.name, total=len(tasks)) for a in adapters}

        def tick(model, done, total):
            prog.update(bars[model], completed=done)

        try:
            result = anyio.run(lambda: run_eval(snap, tasks, adapters, tasks_file=tasks_path, progress=tick))
        except ValueError as exc:
            err.print(f"[red]{exc}[/]")
            sys.exit(2)

    path = save_run(result, out_dir)
    print_run(console, [result], matrices=not no_matrix)
    console.print(f"[dim]Saved run to {path}[/]")
    if html_path:
        write_html([result], html_path)
        console.print(f"[dim]HTML report: {html_path}[/]")
    if fail_under is not None:
        worst = min(m.metrics.accuracy for m in result.models)
        if worst < fail_under:
            err.print(f"[red]Tool accuracy {worst:.0%} is below --fail-under {fail_under:.0%}[/]")
            sys.exit(1)


@main.command()
@server_opt
@tools_opt
@click.option("--tasks", "-t", "tasks_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--min-per-tool", default=1, show_default=True, help="Flag tools with fewer tasks than this.")
def validate(server, tools_file, tasks_path, min_per_tool):
    """Check a task set for label leakage, unknown tools, bad args and duplicates."""
    snap = _snapshot(server, tools_file)
    tasks = _tasks(tasks_path)
    problems = check_tasks(snap, tasks, min_per_tool=min_per_tool)
    for p in problems:
        console.print(f"[yellow]•[/] {p}")
    if problems:
        console.print(f"[red]{len(problems)} problems[/] in {len(tasks)} tasks")
        sys.exit(1)
    console.print(f"[green]OK[/]: {len(tasks)} tasks, {len(snap.tools)} tools, no problems found.")


@main.command()
@click.argument("runs", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--html", "html_path", type=click.Path(dir_okay=False), help="Write an HTML report here.")
@click.option("--no-matrix", is_flag=True, help="Skip confusion matrices in terminal output.")
def report(runs, html_path, no_matrix):
    """Re-render one or more saved runs. Pass two runs to compare them side by side."""
    try:
        loaded = [load_run(r) for r in runs]
    except Exception as exc:
        err.print(f"[red]Could not load run:[/] {exc}")
        sys.exit(2)
    print_run(console, loaded, matrices=not no_matrix)
    if len(loaded) > 1:
        base = loaded[0]
        for other in loaded[1:]:
            for m in other.models:
                try:
                    before = base.model(m.model).metrics.accuracy
                except KeyError:
                    continue
                delta = m.metrics.accuracy - before
                style = "green" if delta > 0 else "red" if delta < 0 else "dim"
                console.print(
                    f"[cyan]{m.model}[/]: {before:.0%} → {m.metrics.accuracy:.0%} "
                    f"([{style}]{delta * 100:+.0f} pts[/]) "
                    f"{base.snapshot.server_name} → {other.snapshot.server_name}"
                )
    if html_path:
        write_html(loaded, html_path)
        console.print(f"[dim]HTML report: {html_path}[/]")


if __name__ == "__main__":  # pragma: no cover
    main()
