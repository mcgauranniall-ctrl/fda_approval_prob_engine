"""Programme event timeline: ClinicalTrials.gov dates + press-release milestones + FDA actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date

from fdaengine.sources import ctgov, openfda
from fdaengine.timeline.press_release_parser import ParsedEvent

# Events whose *earliest* mention is the event; PDUFA_DATE is handled separately (its value can change).
_EARLIEST = {"PHASE2_POSITIVE", "PHASE2_NEGATIVE", "PHASE3_POSITIVE", "PHASE3_NEGATIVE", "NDA_SUBMITTED",
             "NDA_ACCEPTED", "PRIORITY_REVIEW", "STANDARD_REVIEW", "PDUFA_EXTENDED", "ADCOM_SCHEDULED",
             "ADCOM_NOT_PLANNED", "ADCOM_VOTE", "APPROVED", "BTD", "FAST_TRACK", "ORPHAN", "SPA",
             "ACCELERATED_PATHWAY"}
CRL_CYCLE_DAYS = 90         # CRL mentions this close together are one letter; further apart, a new review cycle


@dataclass
class Event:
    event_type: str
    date: str
    source: str                    # 'ctgov' | 'press_release' | 'drugsfda'
    detail: str = ""
    value: str | None = None
    quote: str | None = None
    source_id: str | None = None
    date_basis: str = "explicit"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Timeline:
    events: list[Event] = field(default_factory=list)

    def first(self, *types: str) -> Event | None:
        hits = [e for e in self.events if e.event_type in types]
        return min(hits, key=lambda e: e.date) if hits else None

    def all(self, *types: str) -> list[Event]:
        return sorted((e for e in self.events if e.event_type in types), key=lambda e: e.date)

    def until(self, cutoff: date) -> Timeline:
        return Timeline([e for e in self.events if e.date <= cutoff.isoformat()])

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in sorted(self.events, key=lambda e: e.date)]


def build(trials: list[ctgov.TrialSummary], parsed: list[ParsedEvent], applications: list[dict]) -> Timeline:
    tl = Timeline()
    for t in trials:
        phase = "P3" if t.max_phase >= 3 else ("P2" if t.max_phase >= 2 else "P1")
        label = f"{t.nct_id} ({t.phase_label})"
        if t.start_date:
            tl.events.append(Event(f"{phase}_START", t.start_date.isoformat(), "ctgov", label, source_id=t.nct_id))
        if t.primary_completion_date and t.status in ("COMPLETED", "TERMINATED", "ACTIVE_NOT_RECRUITING"):
            tl.events.append(Event(f"{phase}_PRIMARY_COMPLETION", t.primary_completion_date.isoformat(), "ctgov",
                                   label, source_id=t.nct_id))
        if t.results_first_post_date:
            tl.events.append(Event(f"{phase}_RESULTS_POSTED", t.results_first_post_date.isoformat(), "ctgov",
                                   label, source_id=t.nct_id))
        if t.status == "TERMINATED" and (t.primary_completion_date or t.start_date):
            tl.events.append(Event(f"{phase}_TERMINATED", (t.primary_completion_date or t.start_date).isoformat(),
                                   "ctgov", label, source_id=t.nct_id))

    earliest: dict[str, ParsedEvent] = {}
    for e in sorted(parsed, key=lambda e: (e.event_date, e.date_basis != "explicit")):
        if e.event_type in _EARLIEST:
            earliest.setdefault(e.event_type, e)
        elif e.event_type == "PDUFA_DATE":
            # Keep each distinct announced goal date (extensions produce a new one).
            if not any(x.value == e.value and x.event_type == "PDUFA_DATE" for x in tl.events):
                tl.events.append(Event("PDUFA_DATE", e.event_date, "press_release", value=e.value, quote=e.quote,
                                       source_id=e.source_id, date_basis=e.date_basis))
        elif e.event_type == "CRL":
            last = max((x.date for x in tl.events if x.event_type == "CRL"), default=None)
            if last is None or (date.fromisoformat(e.event_date) - date.fromisoformat(last)).days > CRL_CYCLE_DAYS:
                tl.events.append(Event("CRL", e.event_date, "press_release", quote=e.quote, source_id=e.source_id,
                                       date_basis=e.date_basis))
    for e in earliest.values():
        tl.events.append(Event(e.event_type, e.event_date, "press_release", value=e.value, quote=e.quote,
                               source_id=e.source_id, date_basis=e.date_basis))

    for app in applications:
        for s in openfda.submissions(app):
            if s.status == "AP" and s.status_date and (s.submission_type == "ORIG"
                                                        or (s.class_code or "").upper().startswith("EFFICACY")):
                tl.events.append(Event("FDA_APPROVAL_ACTION", s.status_date, "drugsfda",
                                       f"{s.application_number} {s.submission_type}-{s.submission_number} "
                                       f"({s.class_description or ''}, {s.review_priority or ''})",
                                       source_id=s.application_number))
    return tl
