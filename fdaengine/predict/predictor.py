"""features.json → prediction.json: probabilities, drivers, flags and a confidence rating."""

from __future__ import annotations

from fdaengine.acquire.case_folder import Case
from fdaengine.model import provisional
from fdaengine.predict import flags as flag_rules


def confidence(pred: dict, n_missing: int, n_conflicts: int, provisional_model: bool) -> tuple[str, list[str]]:
    lo, hi = pred["p_approval"]["ci95"]
    width = hi - lo
    reasons = [f"95% interval width {width:.0%}"]
    score = 2 if width < 0.25 else 1 if width < 0.45 else 0
    if n_missing:
        score -= 1
        reasons.append(f"{n_missing} key input(s) missing")
    if n_conflicts:
        score -= 1
        reasons.append(f"{n_conflicts} source conflict(s)")
    if provisional_model and score > 1:
        score = 1
        reasons.append("capped at Medium: provisional (unfitted) model")
    return ["Low", "Medium", "High"][max(0, min(2, score))], reasons


def predict_case(case: Case) -> dict:
    meta = case.read_json("case.json")
    feats = case.read_json("features.json")
    level, features = feats["level"], feats["features"]
    pred = provisional.predict(level, features)
    extraction = case.read_json("extracted/fields.json") or {}
    mflags = flag_rules.model_flags(level, features)
    dflags = flag_rules.data_flags(level, features, feats["conflicts"], extraction.get("rejected", []), meta,
                                   case.read_json("manifest.json", {}))
    missing = [k for k in flag_rules.KEY_FEATURES[level] if k not in features]
    conf, reasons = confidence(pred, len(missing), len(feats["conflicts"]), True)
    contribs = pred["contributions"]
    pred.update({
        "confidence": conf, "confidence_reasons": reasons,
        "drivers_positive": sorted([c for c in contribs if c["logit"] > 0.02], key=lambda c: -c["logit"])[:6],
        "drivers_negative": sorted([c for c in contribs if c["logit"] < -0.02], key=lambda c: c["logit"])[:6],
        "model_flags": mflags, "data_flags": dflags,
        "analyst_flags": case.read_json("extracted/analyst_flags.json", []),
    })
    case.write_json("prediction.json", pred)
    return pred
