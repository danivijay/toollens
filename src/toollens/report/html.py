"""Self-contained HTML report (no external assets), light and dark mode."""

from __future__ import annotations

import html
from pathlib import Path

from toollens import __version__
from toollens.lint import Severity
from toollens.models import ERROR, NO_TOOL
from toollens.runner import ModelRun, Run

# Sequential ramps (blue for correct, orange for mistakes), from light to dark.
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE = ["#fde3d6", "#f9c3a8", "#f4a07a", "#eb6834", "#c94f1f", "#a03d15", "#7a2d0e"]

CSS = """
:root{color-scheme:light;--bg:#f7f7f5;--surface:#fcfcfb;--border:#e4e3df;--text:#0b0b0b;--text-2:#52514e;
--muted:#8a8984;--accent:#2a78d6;--good:#1a7f4b;--warn:#a86400;--bad:#c4322f;--empty:#f0efec;--chip:#efeeea}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#121211;--surface:#1a1a19;
--border:#2e2e2b;--text:#fff;--text-2:#c3c2b7;--muted:#8f8e86;--accent:#3987e5;--good:#3fb37a;--warn:#e0a030;
--bad:#e66767;--empty:#242422;--chip:#262624}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#121211;--surface:#1a1a19;--border:#2e2e2b;--text:#fff;
--text-2:#c3c2b7;--muted:#8f8e86;--accent:#3987e5;--good:#3fb37a;--warn:#e0a030;--bad:#e66767;--empty:#242422;--chip:#262624}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;margin:0 0 4px;letter-spacing:-.01em}h2{font-size:19px;margin:40px 0 12px}
h3{font-size:15px;margin:20px 0 8px;color:var(--text-2)}
.sub{color:var(--text-2);margin:0}.meta{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 0}
.chip{background:var(--chip);border:1px solid var(--border);border-radius:999px;padding:2px 10px;font-size:13px;color:var(--text-2)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px 20px;margin:12px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px}
.tile .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:30px;font-weight:650;font-variant-numeric:tabular-nums;margin-top:2px}
.tile .d{font-size:13px;color:var(--text-2)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--border);vertical-align:top}
th{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;font-weight:600}
td.num,th.num{text-align:right}.scroll{overflow-x:auto}
code{font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--chip);padding:1px 5px;border-radius:5px}
.sev{font-size:12px;font-weight:600;padding:1px 8px;border-radius:999px;border:1px solid currentColor}
.sev-error{color:var(--bad)}.sev-warning{color:var(--warn)}.sev-info{color:var(--muted)}
.bar{height:8px;border-radius:4px;background:var(--empty);position:relative;min-width:80px}
.bar>span{position:absolute;left:0;top:0;bottom:0;border-radius:4px;background:var(--accent)}
.pair{display:grid;grid-template-columns:1fr auto;gap:4px 16px;padding:12px 0;border-bottom:1px solid var(--border)}
.pair:last-child{border-bottom:0}.pair .ex{grid-column:1/-1;color:var(--text-2);font-size:13.5px}
.ok{color:var(--good)}.no{color:var(--bad)}
svg text{fill:var(--text-2);font:12px ui-sans-serif,system-ui,sans-serif}
svg .cell:hover{stroke:var(--text);stroke-width:2}
details summary{cursor:pointer;color:var(--text-2)}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--text-2);margin-top:8px}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin-right:6px}
footer{margin-top:48px;color:var(--muted);font-size:13px}
"""


def e(text: object) -> str:
    return html.escape(str(text), quote=True)


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def _ms(x: float) -> str:
    return "<1 ms" if x < 1 else f"{x:,.0f} ms"


def _ramp(ramp: list[str], value: int, vmax: int) -> str:
    if value <= 0 or vmax <= 0:
        return "var(--empty)"
    idx = min(len(ramp) - 1, max(0, round((value / vmax) * (len(ramp) - 1))))
    return ramp[idx]


def confusion_svg(m: ModelRun) -> str:
    labels = [lbl for lbl in m.metrics.labels]
    rows = [lbl for lbl in labels if lbl != ERROR]
    conf = m.metrics.confusion
    n_rows, n_cols = len(rows), len(labels)
    cell, gap = 40, 2
    left, top = 150, 112
    width = left + n_cols * cell + 16
    height = top + n_rows * cell + 16
    row_tot = {r: sum(conf.get(r, {}).values()) for r in rows}
    vmax_ok = max([conf.get(r, {}).get(r, 0) for r in rows] + [1])
    vmax_bad = max([v for r in rows for c, v in conf.get(r, {}).items() if c != r] + [1])

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
        f'aria-label="Confusion matrix for {e(m.model)}">'
    ]
    for j, c in enumerate(labels):
        x = left + j * cell + cell / 2
        parts.append(
            f'<text transform="translate({x + 4},{top - 8}) rotate(-50)" text-anchor="start">{e(c)}</text>'
        )
    for i, r in enumerate(rows):
        y = top + i * cell
        parts.append(f'<text x="{left - 10}" y="{y + cell / 2 + 4}" text-anchor="end">{e(r)}</text>')
        for j, c in enumerate(labels):
            v = conf.get(r, {}).get(c, 0)
            diag = r == c
            fill = _ramp(BLUE if diag else ORANGE, v, vmax_ok if diag else vmax_bad)
            share = f"{v / row_tot[r] * 100:.0f}%" if row_tot[r] else "0%"
            tip = f"expected {r} → predicted {c}: {v} of {row_tot[r]} ({share})"
            x = left + j * cell
            parts.append(
                f'<rect class="cell" x="{x + gap / 2}" y="{y + gap / 2}" width="{cell - gap}" '
                f'height="{cell - gap}" rx="4" fill="{fill}"><title>{e(tip)}</title></rect>'
            )
            if v:
                ramp = BLUE if diag else ORANGE
                dark = fill in ramp[3:]
                color = "#ffffff" if dark else "#0b0b0b"
                parts.append(
                    f'<text x="{x + cell / 2}" y="{y + cell / 2 + 4}" text-anchor="middle" '
                    f'style="fill:{color};font-weight:600" pointer-events="none">{v}</text>'
                )
    parts.append(
        f'<text x="4" y="{top - 10}" text-anchor="start" style="fill:var(--muted)">expected ↓ · predicted →</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _tiles(run: Run, m: ModelRun) -> str:
    mt = m.metrics
    k = mt.accuracy_by_kind
    c = mt.count_by_kind
    tiles = [
        ("Tool accuracy", pct(mt.accuracy), f"{sum(r.tool_correct for r in m.results)} of {mt.n} tasks"),
        ("Direct", pct(k.get("direct")), f"{c.get('direct', 0)} plain requests"),
        ("Near-miss", pct(k.get("near_miss")), f"{c.get('near_miss', 0)} requests luring a sibling"),
        ("Abstains correctly", pct(mt.abstention_recall), f"{c.get('no_tool', 0)} out-of-scope requests"),
    ]
    if m.produces_arguments:
        tiles.append(("Args valid", pct(mt.args_valid_rate), "against the tool's JSON Schema"))
        tiles.append(("Args match", pct(mt.args_match_rate), "contain the expected values"))
    return (
        '<div class="tiles">'
        + "".join(
            f'<div class="tile"><div class="k">{e(a)}</div><div class="v">{e(b)}</div><div class="d">{e(d)}</div></div>'
            for a, b, d in tiles
        )
        + "</div>"
    )


def _pairs(run: Run, m: ModelRun, limit: int = 6) -> str:
    tasks = {r.task.id: r for r in m.results}
    out = []
    for p in m.metrics.confused_pairs[:limit]:
        examples = []
        for tid in p.examples[:2]:
            r = tasks[tid]
            examples.append(
                f"“{e(r.task.prompt)}” <span class='no'>→ {e(r.predicted_label)}</span>"
                f" (expected <code>{e(r.expected_label)}</code>)"
            )
        out.append(
            f'<div class="pair"><div><code>{e(p.a)}</code> ⇄ <code>{e(p.b)}</code></div>'
            f'<div><b>{p.total}</b> mistakes <span class="sub">({p.a_as_b} → / {p.b_as_a} ←)</span></div>'
            f'<div class="ex">{"<br>".join(examples)}</div></div>'
        )
    mt = m.metrics
    extra = (
        f"<p class='sub'>Also: <b>{mt.missed_calls}</b> missed calls (needed a tool, called none) · "
        f"<b>{mt.false_calls}</b> false calls (out of scope, called a tool anyway) · <b>{mt.errors}</b> errors</p>"
    )
    return (
        '<div class="card">'
        + ("".join(out) or "<p class='sub'>No tool-to-tool confusions.</p>")
        + extra
        + "</div>"
    )


def _per_tool(m: ModelRun) -> str:
    rows = []
    for s in sorted(m.metrics.per_tool, key=lambda s: s.recall if s.recall is not None else 2):
        rec = s.recall or 0
        rows.append(
            f"<tr><td><code>{e(s.tool)}</code></td><td class='num'>{s.support}</td>"
            f"<td class='num'>{pct(s.precision)}</td><td class='num'>{pct(s.recall)}</td>"
            f"<td><div class='bar' title='recall {pct(s.recall)}'><span style='width:{rec * 100:.0f}%'></span></div></td></tr>"
        )
    return (
        '<div class="card scroll"><table><thead><tr><th>Tool</th><th class="num">Tasks</th>'
        '<th class="num">Precision</th><th class="num">Recall</th><th>Recall</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _failures(m: ModelRun) -> str:
    rows = [
        f"<tr><td><code>{e(r.task.id)}</code></td><td>{e(r.task.prompt)}</td>"
        f"<td><code>{e(r.expected_label)}</code></td><td class='no'><code>{e(r.predicted_label)}</code>"
        f"{('<br><span class=sub>' + e(r.prediction.error) + '</span>') if r.prediction.error else ''}</td></tr>"
        for r in m.results
        if not r.tool_correct
    ]
    if not rows:
        return ""
    return (
        f"<details><summary>All {len(rows)} wrong tool choices</summary><div class='card scroll'><table>"
        "<thead><tr><th>Task</th><th>Prompt</th><th>Expected</th><th>Called</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></details>"
    )


def _lint(run: Run) -> str:
    lr = run.lint
    rows = "".join(
        f"<tr><td><span class='sev sev-{f.severity.value}'>{e(f.severity.value)}</span></td>"
        f"<td><code>{e(f.code)}</code></td><td>{', '.join(f'<code>{e(t)}</code>' for t in f.tools)}</td>"
        f"<td>{e(f.message)}</td></tr>"
        for f in lr.findings
    )
    summary = (
        f"{lr.count(Severity.ERROR)} errors · {lr.count(Severity.WARNING)} warnings · "
        f"{lr.count(Severity.INFO)} notes · tool definitions ≈{lr.total_tokens:,} tokens"
    )
    return (
        f"<p class='sub'>{e(summary)}</p><div class='card scroll'><table><thead><tr><th>Severity</th><th>Code</th>"
        f"<th>Tools</th><th>Finding</th></tr></thead><tbody>{rows or '<tr><td colspan=4>Clean.</td></tr>'}"
        "</tbody></table></div>"
    )


def _scoreboard(runs: list[Run]) -> str:
    rows = []
    for run in runs:
        for m in run.models:
            k = m.metrics.accuracy_by_kind
            rows.append(
                f"<tr><td>{e(run.snapshot.server_name or run.snapshot.server)}</td><td><code>{e(m.model)}</code></td>"
                f"<td class='num'><b>{pct(m.metrics.accuracy)}</b></td><td class='num'>{pct(k.get('direct'))}</td>"
                f"<td class='num'>{pct(k.get('near_miss'))}</td><td class='num'>{pct(m.metrics.abstention_recall)}</td>"
                f"<td class='num'>{pct(m.metrics.args_valid_rate)}</td><td class='num'>{pct(m.metrics.args_match_rate)}</td>"
                f"<td class='num'>{m.metrics.errors}</td><td class='num'>{_ms(m.metrics.mean_latency_ms)}</td></tr>"
            )
    return (
        "<div class='card scroll'><table><thead><tr><th>Server</th><th>Model</th><th class='num'>Accuracy</th>"
        "<th class='num'>Direct</th><th class='num'>Near-miss</th><th class='num'>Abstain</th>"
        "<th class='num'>Args valid</th><th class='num'>Args match</th><th class='num'>Errors</th>"
        "<th class='num'>Latency</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def render_html(runs: list[Run]) -> str:
    first = runs[0]
    title = "ToolLens report"
    servers = " vs ".join(e(r.snapshot.server_name or r.snapshot.server) for r in runs)
    body = [
        f"<h1>{title}</h1><p class='sub'>{servers}</p>",
        "<div class='meta'>"
        + "".join(
            f"<span class='chip'>{e(r.snapshot.server_name or 'server')} · {len(r.snapshot.tools)} tools · "
            f"protocol {e(r.snapshot.protocol_version or '?')} · ≈{r.lint.total_tokens:,} tokens</span>"
            for r in runs
        )
        + f"<span class='chip'>{first.models[0].metrics.n if first.models else 0} tasks</span>"
        f"<span class='chip'>{e(first.created)}</span></div>",
        "<h2>Scoreboard</h2>",
        _scoreboard(runs),
    ]
    for run in runs:
        sname = e(run.snapshot.server_name or run.snapshot.server)
        for m in run.models:
            body += [
                f"<h2>{sname} · <code>{e(m.model)}</code></h2>",
                _tiles(run, m),
                "<h3>Confusion matrix</h3>",
                "<div class='card scroll'>" + confusion_svg(m) + "<div class='legend'>"
                f"<span><span class='sw' style='background:{BLUE[4]}'></span>correct (diagonal)</span>"
                f"<span><span class='sw' style='background:{ORANGE[3]}'></span>wrong tool, abstained or errored</span>"
                f"<span>Rows are what the task needed; <code>{e(NO_TOOL)}</code> means no tool call.</span>"
                "</div></div>",
                "<h3>Most confused pairs</h3>",
                _pairs(run, m),
                "<h3>Per-tool precision and recall</h3>",
                _per_tool(m),
                _failures(m),
            ]
        body += [f"<h2>Static lint · {sname}</h2>", _lint(run)]
    body.append(
        f"<footer>Generated by ToolLens {e(__version__)}. Token counts are estimates (≈4 characters per token).</footer>"
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title><style>{CSS}</style></head><body><main>"
        + "".join(body)
        + "</main></body></html>"
    )


def write_html(runs: list[Run], path: str | Path) -> Path:
    p = Path(path)
    p.write_text(render_html(runs), encoding="utf-8")
    return p
