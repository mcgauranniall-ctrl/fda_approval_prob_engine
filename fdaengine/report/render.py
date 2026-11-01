"""Render the v1-style HTML report (self-contained: CSS inlined, charts as inline SVG)."""

from __future__ import annotations

import html
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from fdaengine import config
from fdaengine.acquire.case_folder import Case
from fdaengine.timeline.levels import LEVELS, ORDER

_STATIC = Path(__file__).parent / "static"


def _pct(x: float | None, digits: int = 0) -> str:
    return "—" if x is None else f"{x * 100:.{digits}f}%"


def path_svg(path: list[dict]) -> Markup:
    if len(path) < 2:
        return Markup("")
    w, h, pad_l, pad_r, pad_t, pad_b = 760, 220, 90, 70, 16, 46
    xs = [pad_l + i * (w - pad_l - pad_r) / (len(path) - 1) for i in range(len(path))]
    y = lambda p: pad_t + (1 - p) * (h - pad_t - pad_b)
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="Probability path across levels">']
    for g in (0, 0.25, 0.5, 0.75, 1):
        parts.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{y(g):.1f}" y2="{y(g):.1f}" stroke="#ddd8cc" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{y(g) + 3:.1f}" text-anchor="end">{int(g * 100)}%</text>')
    band = " ".join(f"{x:.1f},{y(p['ci95'][1]):.1f}" for x, p in zip(xs, path)) + " " + \
           " ".join(f"{x:.1f},{y(p['ci95'][0]):.1f}" for x, p in reversed(list(zip(xs, path))))
    parts.append(f'<polygon points="{band}" fill="#c8401a" fill-opacity="0.10"/>')
    parts.append('<polyline fill="none" stroke="#c8401a" stroke-width="2" points="'
                 + " ".join(f"{x:.1f},{y(p['p']):.1f}" for x, p in zip(xs, path)) + '"/>')
    for x, p in zip(xs, path):
        parts.append(f'<circle cx="{x:.1f}" cy="{y(p["p"]):.1f}" r="4" fill="#0f0f0f"/>')
        parts.append(f'<text x="{x:.1f}" y="{y(p["p"]) - 9:.1f}" text-anchor="middle" style="fill:#0f0f0f;font-weight:500">'
                     f'{p["p"] * 100:.0f}%</text>')
        parts.append(f'<text x="{x:.1f}" y="{h - pad_b + 16}" text-anchor="middle">{p["level"]} · {html.escape(p["name"][:18])}</text>')
        parts.append(f'<text x="{x:.1f}" y="{h - pad_b + 30}" text-anchor="middle">{p["cutoff"]}</text>')
    parts.append("</svg>")
    return Markup("".join(parts))


def histogram_svg(counts: list[int], median: float) -> Markup:
    w, h, pad = 760, 120, 24
    top = max(counts) or 1
    bw = (w - 2 * pad) / len(counts)
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="Distribution of simulated probabilities">']
    for i, c in enumerate(counts):
        bh = (h - 2 * pad) * c / top
        parts.append(f'<rect x="{pad + i * bw + 1:.1f}" y="{h - pad - bh:.1f}" width="{bw - 2:.1f}" height="{bh:.1f}" fill="#0f0f0f" fill-opacity="0.75"/>')
    mx = pad + median * (w - 2 * pad)
    parts.append(f'<line x1="{mx:.1f}" x2="{mx:.1f}" y1="{pad - 6}" y2="{h - pad}" stroke="#c8401a" stroke-width="2"/>')
    for g in (0, 0.25, 0.5, 0.75, 1):
        parts.append(f'<text x="{pad + g * (w - 2 * pad):.1f}" y="{h - 6}" text-anchor="middle">{int(g * 100)}%</text>')
    parts.append("</svg>")
    return Markup("".join(parts))


def memo_text(meta: dict, pred: dict, path: list[dict], feats: dict) -> str:
    p = pred["p_approval"]
    lines = [
        f"FDA APPROVAL PROBABILITY ASSESSMENT — {meta['drug']} | {meta['indication']}",
        f"Level {meta['level']} ({meta['level_name']}), information cut-off {meta['cutoff']}",
        "─" * 66,
        f"P(programme ends in FDA approval): {_pct(p['median'], 1)}  (95% interval {_pct(p['ci95'][0])}–{_pct(p['ci95'][1])})",
    ]
    if "p_next_action" in pred:
        n = pred["p_next_action"]
        lines.append(f"P(approval at the next FDA action): {_pct(n['median'], 1)}  "
                     f"(95% interval {_pct(n['ci95'][0])}–{_pct(n['ci95'][1])})")
    lines += [f"Confidence: {pred['confidence']} — " + "; ".join(pred["confidence_reasons"]),
              f"Base rate for programmes at this level: {_pct(pred['base_rate'])}", ""]
    if len(path) > 1:
        lines.append("PROBABILITY PATH:")
        lines += [f"  {x['level']} {x['name']:<26} {x['cutoff']}  {_pct(x['p'])}" for x in path]
        lines.append("")
    lines.append("WHAT MOVES THE ESTIMATE (log-odds vs base rate):")
    for c in sorted(pred["contributions"], key=lambda c: -abs(c["logit"]))[:8]:
        lines.append(f"  {c['logit']:+.2f}  {c['label']}")
    crit = [f["text"] for f in pred["model_flags"] if f["severity"] == "critical"]
    lines += ["", "CRITICAL FLAGS:"] + ([f"  - {t}" for t in crit] or ["  (none)"])
    lines += ["", "MODEL: " + pred["model_version"],
              "The probabilities come from literature-informed priors, not a model fitted to historical outcomes.",
              "Treat them as a structured starting point, not a calibrated forecast.",
              "", "Not investment, actuarial or medical advice."]
    return "\n".join(lines)


def render(case: Case) -> Path:
    meta = case.read_json("case.json")
    pred = case.read_json("prediction.json")
    feats = case.read_json("features.json")
    path = case.read_json("analysis/path.json", [])
    extraction = case.read_json("extracted/fields.json") or {}
    side = case.root.with_name(case.root.name + ".meta.json")
    side_meta = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
    timeline = [e for e in case.read_json("timeline.json", []) if not e["event_type"].startswith("P1_")]

    env = Environment(loader=FileSystemLoader(config.TEMPLATES_DIR), autoescape=select_autoescape(["html", "j2"]))
    env.filters["pct"] = _pct
    tpl = env.get_template("report.html.j2")
    p = pred["p_approval"]["median"]
    max_logit = max([abs(c["logit"]) for c in pred["contributions"]] + [0.5])
    out = tpl.render(
        meta=meta, pred=pred, feats=feats["features"], conflicts=feats["conflicts"], path=path,
        extraction=extraction, timeline=timeline, levels=LEVELS, order=ORDER,
        reached={lv["level"]: lv for lv in side_meta.get("levels", [])},
        outcome=side_meta.get("actual_outcome"),
        css=Markup((_STATIC / "report.css").read_text(encoding="utf-8")),
        prob_class="high" if p >= 0.6 else "medium" if p >= 0.4 else "low",
        path_svg=path_svg(path), hist_svg=histogram_svg(pred["draws_histogram"], p),
        max_logit=max_logit, memo=memo_text(meta, pred, path, feats["features"]),
    )
    target = case.path("report", "report.html")
    target.write_text(out, encoding="utf-8")
    return target
