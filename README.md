# FDA Approval Probability Engine (v2)

Forecasts **P(a drug's Phase 3 programme ends in FDA approval)** at a milestone level you choose: Phase 2
complete, Phase 3 readout, Filed, Post-AdCom, Pre-PDUFA, Resubmission, or Auto. It only uses information public
before that level's cut-off date. Design and roadmap: [PLAN.md](PLAN.md). The original v1 web app is in
[legacy/](legacy/webapp_v1.html).

## Setup

```powershell
uv sync                      # Python 3.12 environment
copy .env.example .env       # then set SEC_USER_AGENT="Your Name you@example.com"
```

`SEC_USER_AGENT` is required for press releases: SEC EDGAR asks for contact details in the User-Agent.
Claude extraction uses the Claude Code CLI (`claude.exe`) on your subscription. No API key is needed.

## Run

Double-click **`launch.bat`**. It sets up the environment on first run, checks `.env`, and opens the UI at
http://localhost:8501. Or from a terminal:

```powershell
uv run streamlit run app/streamlit_app.py          # UI
uv run fdaengine-case --drug suzetrigine --indication "moderate to severe acute pain" --level L2 --as-of 2026-10-29
uv run pytest                                      # unit tests (offline)
```

The UI flow:
1. Enter the drug and indication. A past as-of date runs a backtest.
2. Confirm the trials that belong to the programme.
3. Pick a level from the ones the programme has reached.
4. Run. Downloads go to `cases/<drug>_<indication>_<level>_<date>/`, a headless `claude -p` run extracts cited
   facts from them, and the report appears in `report/report.html`.

## What's built (P0–P2, P7, P8 slice)

| Stage | Module |
|---|---|
| Resolve names, candidate trials, FDA applications, SEC filer | `fdaengine/resolve/programme.py` |
| Rule-based press-release milestone parser (no LLM) | `fdaengine/timeline/press_release_parser.py` |
| Programme timeline + level cut-offs | `fdaengine/timeline/events.py`, `levels.py` |
| Acquisition with cut-off enforcement + manifest | `fdaengine/pipeline.py`, `acquire/` |
| Structured ClinicalTrials.gov stats (p→z, SAE RR, Newcombe, Fisher, assurance) | `prepare/ctgov_structured.py`, `features/stats_transforms.py` |
| Headless Claude extraction (read-only tools, JSON schema, verified quotes) | `agent/claude_runner.py`, `agent/extract.py`, `prompts/extract_case.md` |
| Features with source priority + conflicts | `features/build.py` |
| **Provisional** prior-only model, flags, confidence, probability path | `model/provisional.py`, `predict/` |
| v1-style HTML report | `report/` |
| Streamlit UI, CLI | `app/streamlit_app.py`, `fdaengine/cli.py` |

**The model is provisional:** literature-informed priors with Monte Carlo uncertainty, not yet fitted to
historical outcomes. The training pipeline (PLAN.md P3–P5) is next.
