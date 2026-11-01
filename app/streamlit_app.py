"""Streamlit UI: `uv run streamlit run app/streamlit_app.py`."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import streamlit as st

from fdaengine import config, pipeline
from fdaengine.resolve.programme import resolve
from fdaengine.timeline.levels import LEVELS

st.set_page_config(page_title="FDA Approval Engine", page_icon="⚖️", layout="wide")
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700;900&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet">
<style>
  /* Colours come from .streamlit/config.toml (pinned light theme); only v1's typographic accents live here. */
  h1 { font-weight: 900 !important; letter-spacing: -0.02em; }
  .step { font-family: 'DM Mono', monospace; font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: #c8401a; }
  .muted { font-family: 'DM Mono', monospace; font-size: 12px; color: #8a8070; }
</style>""", unsafe_allow_html=True)

ss = st.session_state
for key, default in (("res", None), ("prog", None), ("case", None), ("status", None)):
    ss.setdefault(key, default)

# ── Sidebar: case library ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Case library")
    cases = sorted((p for p in config.CASES_DIR.glob("*") if p.is_dir() and (p / "report" / "report.html").exists()),
                   key=lambda p: p.stat().st_mtime, reverse=True) if config.CASES_DIR.exists() else []
    if cases:
        pick = st.selectbox("Open a previous report", ["—"] + [p.name for p in cases])
        if pick != "—":
            ss.case = str(config.CASES_DIR / pick)
    else:
        st.caption("No cases yet.")
    st.markdown("---")
    st.caption("Claude CLI: " + ("found" if config.claude_exe() else "**not found**"))
    st.caption("SEC_USER_AGENT: " + ("set" if config.sec_user_agent() else "**not set** — press releases disabled"))

st.markdown('<div class="step">Quantitative regulatory analysis · v2</div>', unsafe_allow_html=True)
st.title("FDA Approval Probability Engine")

# ── Step 1: drug ──────────────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="step">Step 1 · Drug</div>', unsafe_allow_html=True)
with st.form("drug"):
    c1, c2 = st.columns(2)
    drug = c1.text_input("Drug (generic, brand or code name)", placeholder="e.g. suzetrigine")
    indication = c2.text_input("Indication", placeholder="e.g. moderate to severe acute pain")
    c3, c4 = st.columns(2)
    extra = c3.text_input("Extra NCT IDs (optional, comma-separated)")
    as_of = c4.date_input("As-of date — today = live forecast, a past date = backtest", value=date.today(),
                          max_value=date.today())
    if st.form_submit_button("Find programme →", type="primary") and drug and indication:
        with st.spinner("Searching ClinicalTrials.gov, openFDA, RxNorm and SEC EDGAR…"):
            ncts = [x.strip().upper() for x in extra.split(",") if x.strip()]
            ss.res, ss.prog, ss.case, ss.status = resolve(drug, indication, extra_ncts=ncts), None, None, None
            ss.as_of, ss.mode = as_of, "backtest" if as_of < date.today() else "live"

# ── Step 2: confirm trials ────────────────────────────────────────────────────────────────────────────
if ss.res:
    res = ss.res
    st.markdown('<div class="step">Step 2 · Confirm the programme</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="muted">Names: {", ".join(res.names)} · Lead sponsor: {res.sponsor or "unknown"}'
                f'{" · SEC filer " + (res.ticker or res.cik) if res.cik else ""} · '
                f'FDA applications: {", ".join(a["application_number"] for a in res.applications) or "none"}</div>',
                unsafe_allow_html=True)
    for w in res.warnings:
        st.warning(w)
    rows = [{"include": c.suggested, "nct": c.trial.nct_id, "phase": c.trial.phase_label, "title": c.trial.title,
             "sponsor": c.trial.sponsor, "conditions": "; ".join(c.trial.conditions)[:80],
             "start": c.trial.start_date, "status": c.trial.status, "why": c.reason}
            for c in res.candidates if 0 < c.trial.max_phase < 4]
    edited = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch",
                            disabled=[c for c in rows[0] if c != "include"] if rows else [],
                            column_config={"include": st.column_config.CheckboxColumn("✓", width="small")})
    if st.button("Build timeline →", type="primary"):
        selected = edited.loc[edited["include"], "nct"].tolist()
        with st.status("Building the programme timeline…", expanded=True) as s:
            ss.prog = pipeline.discover(res, selected, lambda m: s.write(m))
            s.update(label="Timeline ready", state="complete", expanded=False)
        ss.case, ss.status = None, None

# ── Step 3: level + run ───────────────────────────────────────────────────────────────────────────────
if ss.prog:
    prog = ss.prog
    st.markdown('<div class="step">Step 3 · Choose the forecast level</div>', unsafe_allow_html=True)
    levels = prog.levels(ss.as_of)
    lv_rows = [{"level": lv.level, "name": lv.name, "reached": "✓" if lv.reached else "—",
                "information cut-off": lv.cutoff or "", "basis": lv.basis + (" (estimated)" if lv.estimated else "")}
               for lv in levels]
    st.dataframe(pd.DataFrame(lv_rows), hide_index=True, width="stretch")
    with st.expander("Programme timeline (all events found)"):
        tl = pd.DataFrame(prog.timeline.to_list())
        if not tl.empty:
            st.dataframe(tl[["date", "event_type", "source", "value", "detail", "quote"]], hide_index=True,
                         width="stretch")
    reached = [lv.level for lv in levels if lv.reached]
    options = ["Auto"] + reached
    level = st.radio("Level", options, horizontal=True,
                     format_func=lambda x: "Auto (latest available)" if x == "Auto" else f"{x} · {LEVELS[x]}")
    c1, c2 = st.columns(2)
    use_claude = c1.checkbox("Run headless Claude extraction (uses your Claude subscription)", value=True)
    model = c2.selectbox("Claude model", ["opus", "sonnet"], index=0, disabled=not use_claude)
    if not reached:
        st.info("No level has been reached yet (no completed Phase 2 found).")
    elif st.button("Run analysis →", type="primary"):
        with st.status("Running the analysis…", expanded=True) as s:
            try:
                case = pipeline.acquire(prog, level, ss.as_of, ss.mode, lambda m: s.write(m))
                ss.status = pipeline.analyse(case, use_claude=use_claude, model=model, progress=lambda m: s.write(m))
                ss.case = str(case.root)
                s.update(label="Analysis complete", state="complete", expanded=False)
            except ValueError as e:
                s.update(label=str(e), state="error")

# ── Step 4: report ────────────────────────────────────────────────────────────────────────────────────
if ss.case:
    from pathlib import Path
    root = Path(ss.case)
    report = root / "report" / "report.html"
    if ss.status and ss.status.get("claude_error"):
        st.error("Claude extraction failed, so the report uses structured data only. " + ss.status["claude_error"])
    if report.exists():
        st.markdown('<div class="step">Report</div>', unsafe_allow_html=True)
        pred = json.loads((root / "prediction.json").read_text(encoding="utf-8"))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("P(approval)", f"{pred['p_approval']['median']:.0%}")
        m2.metric("95% interval", f"{pred['p_approval']['ci95'][0]:.0%} – {pred['p_approval']['ci95'][1]:.0%}")
        m3.metric("Confidence", pred["confidence"])
        m4.metric("Level", pred["level"])
        html = report.read_text(encoding="utf-8")
        st.download_button("Download report (HTML)", html, file_name=f"{root.name}.html", mime="text/html")
        st.iframe(report, height=2600)   # our own template; untrusted text is escaped and scripts are blocked by CSP
        st.caption(f"Case folder: {root}")
