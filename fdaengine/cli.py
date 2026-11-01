"""Command-line equivalent of the UI: `uv run fdaengine-case --drug suzetrigine --indication "acute pain"`."""

from __future__ import annotations

import argparse
import sys
from datetime import date

from fdaengine import pipeline
from fdaengine.resolve.programme import resolve
from fdaengine.timeline.levels import ORDER


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FDA approval probability for a drug × indication")
    ap.add_argument("--drug", required=True)
    ap.add_argument("--indication", required=True)
    ap.add_argument("--level", default="Auto", choices=["Auto", *ORDER])
    ap.add_argument("--as-of", default=None, help="backtest date YYYY-MM-DD (default: today, live mode)")
    ap.add_argument("--nct", nargs="*", default=[], help="extra NCT IDs to include")
    ap.add_argument("--no-claude", action="store_true", help="skip the headless Claude extraction")
    ap.add_argument("--model", default="opus")
    args = ap.parse_args(argv)

    say = lambda m: print(f"  · {m}", flush=True)
    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()
    mode = "backtest" if args.as_of else "live"
    print(f"Resolving {args.drug} / {args.indication}…")
    res = resolve(args.drug, args.indication, extra_ncts=args.nct)
    selected = [c.trial.nct_id for c in res.candidates if c.suggested] + args.nct
    for c in res.candidates:
        if c.trial.nct_id in selected:
            print(f"  ✓ {c.trial.nct_id} {c.trial.phase_label:<6} {c.trial.title[:70]}")
    for w in res.warnings:
        print(f"  ! {w}")
    prog = pipeline.discover(res, selected, say)
    for lv in prog.levels(as_of):
        print(f"  {lv.level} {lv.name:<26} {'reached' if lv.reached else '—':<8} {lv.cutoff or ''}  {lv.basis}")
    try:
        case = pipeline.acquire(prog, args.level, as_of, mode, say)
    except ValueError as e:
        print(f"Cannot run: {e}")
        return 1
    status = pipeline.analyse(case, use_claude=not args.no_claude, model=args.model, progress=say)
    pred = case.read_json("prediction.json")
    p = pred["p_approval"]
    print(f"\nP(approval) = {p['median']:.0%}  (95% interval {p['ci95'][0]:.0%}–{p['ci95'][1]:.0%}), "
          f"confidence {pred['confidence']}  [{pred['model_version']}]")
    if status["claude_error"]:
        print(f"Claude extraction failed: {status['claude_error']}")
    print(f"Report: {status['report']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
