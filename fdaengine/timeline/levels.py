"""Forecast levels (landmarks) and their information cut-off dates. See PLAN.md §2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta

from fdaengine.timeline.events import Timeline

LEVELS = {
    "L1": "Phase 2 complete",
    "L2": "Phase 3 readout",
    "L3": "Filed (NDA/BLA accepted)",
    "L4": "Post-AdCom",
    "L5": "Pre-PDUFA",
    "L6": "Resubmission after CRL",
}
ORDER = list(LEVELS)


@dataclass
class LevelStatus:
    level: str
    name: str
    cutoff: str | None          # last date whose information may be used
    reached: bool               # cutoff <= as_of and before the decision
    basis: str                  # how the cut-off was determined
    estimated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _d(s: str | None) -> date | None:
    return date.fromisoformat(s) if s else None


def decision(tl: Timeline) -> tuple[str | None, date | None]:
    """('APPROVED' | 'CRL' | None, date) of the programme's first FDA decision."""
    approval = tl.first("APPROVED")
    accepted = tl.first("NDA_ACCEPTED", "PRIORITY_REVIEW", "STANDARD_REVIEW", "PDUFA_DATE")
    readout = tl.first("PHASE3_POSITIVE", "PHASE3_NEGATIVE")
    floor = (accepted or readout).date if (accepted or readout) else "0000"
    fda = [e for e in tl.all("FDA_APPROVAL_ACTION") if e.date >= floor]
    crl = tl.first("CRL")
    candidates = []
    if approval:
        # Prefer the Drugs@FDA action date within 45 days of the press-release mention.
        near = [e for e in fda if abs((_d(e.date) - _d(approval.date)).days) <= 45]
        candidates.append(("APPROVED", _d(near[0].date if near else approval.date)))
    elif fda and readout:
        candidates.append(("APPROVED", _d(fda[0].date)))
    if crl:
        candidates.append(("CRL", _d(crl.date)))
    if not candidates:
        return None, None
    return min(candidates, key=lambda c: c[1])


def compute(tl: Timeline, as_of: date) -> list[LevelStatus]:
    out: list[LevelStatus] = []
    _, decided = decision(tl)

    def add(level, cutoff: date | None, basis: str, estimated: bool = False):
        ok = cutoff is not None and cutoff <= as_of and (decided is None or cutoff < decided)
        out.append(LevelStatus(level, LEVELS[level], cutoff.isoformat() if cutoff else None, ok, basis, estimated))

    # L1: the day before the earliest sign of Phase 3 results (topline release or primary completion).
    readout = tl.first("PHASE3_POSITIVE", "PHASE3_NEGATIVE")
    p3_pc = tl.first("P3_PRIMARY_COMPLETION")
    p2_done = tl.first("P2_PRIMARY_COMPLETION", "PHASE2_POSITIVE", "PHASE2_NEGATIVE")
    bounds = [_d(e.date) for e in (readout, p3_pc) if e]
    if p2_done and bounds:
        add("L1", min(bounds) - timedelta(days=1), "day before first Phase 3 primary completion / topline")
    elif p2_done:
        add("L1", as_of, "Phase 2 complete; no Phase 3 results yet")
    else:
        add("L1", None, "no completed Phase 2 found")

    # L2: day after the Phase 3 topline release (estimated from primary completion if no release was found).
    if readout:
        add("L2", _d(readout.date) + timedelta(days=1), f"day after Phase 3 topline ({readout.source})")
    elif p3_pc:
        last_pc = max(_d(e.date) for e in tl.all("P3_PRIMARY_COMPLETION"))
        add("L2", last_pc + timedelta(days=90), "Phase 3 primary completion + 90 days (no topline release found)", True)
    else:
        add("L2", None, "Phase 3 not read out")

    accepted = tl.first("NDA_ACCEPTED", "PRIORITY_REVIEW", "STANDARD_REVIEW", "PDUFA_DATE")
    add("L3", _d(accepted.date) + timedelta(days=1) if accepted else None,
        "day after filing acceptance announced" if accepted else "no filing acceptance found")

    vote = tl.first("ADCOM_VOTE")
    if vote:
        add("L4", _d(vote.date) + timedelta(days=1), "day after AdCom vote")
    else:
        none_planned = tl.first("ADCOM_NOT_PLANNED")
        add("L4", None, "FDA said no AdCom planned" if none_planned else "no AdCom found")

    pdufas = tl.all("PDUFA_DATE")
    if pdufas:
        goal = max(_d(e.value) for e in pdufas if e.value)
        add("L5", goal - timedelta(days=7), f"7 days before PDUFA goal date {goal.isoformat()}")
    else:
        add("L5", None, "no PDUFA date found")

    crl = tl.first("CRL")
    resub = [e for e in tl.all("NDA_ACCEPTED", "PDUFA_DATE") if crl and e.date > crl.date]
    add("L6", _d(resub[0].date) + timedelta(days=1) if resub else None,
        "day after resubmission accepted" if resub else "no CRL / resubmission")
    return out


def auto_level(levels: list[LevelStatus]) -> LevelStatus | None:
    reached = [lv for lv in levels if lv.reached]
    return max(reached, key=lambda lv: ORDER.index(lv.level)) if reached else None
