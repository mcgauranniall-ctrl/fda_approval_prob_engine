"""Rule-based extraction of regulatory/clinical milestones from press releases.

This parser is what builds the historical training timelines (no LLM), so it is deliberately conservative:
an event is only emitted when the sentence names the event explicitly, refers to this drug (in the sentence
or the immediately preceding context), and does not attribute it to a different indication.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import asdict, dataclass
from datetime import date

from rapidfuzz import fuzz

from fdaengine.dates import parse_partial

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
_MONTH_RE = "|".join(m.capitalize() for m in MONTHS)
DATE_FULL = re.compile(rf"\b({_MONTH_RE})\s+(\d{{1,2}}),?\s+(\d{{4}})\b")
DATE_MONTH_YEAR = re.compile(rf"\b(?:[Ii]n|[Oo]n|[Bb]y)\s+({_MONTH_RE})(?:\s+(\d{{4}}))?\b")
DATE_QUARTER = re.compile(r"\b(first|second|third|fourth)\s+quarter\s+of\s+(\d{4})\b", re.IGNORECASE)
FUTURE = re.compile(r"\b(plans? to|expects? to|anticipates?|anticipated|on track to|will|intends? to|"
                    r"potential(ly)?|may|could|upcoming|planned|target(s|ing)? (to|for)|by mid-|aims? to)\b", re.IGNORECASE)
# Multi-drug documents (pipeline tables, earnings releases) put many drugs near each event phrase: the drug must be
# named close to the phrase, or in the preceding sentences of a short, ordinary sentence.
MAX_NAME_DISTANCE = 250
MAX_CONTEXT_SENTENCE = 300
REGULATORY = {"NDA_SUBMITTED", "NDA_ACCEPTED", "PRIORITY_REVIEW", "STANDARD_REVIEW", "PDUFA_DATE", "PDUFA_EXTENDED",
              "ADCOM_NOT_PLANNED", "ADCOM_VOTE", "ADCOM_SCHEDULED", "CRL", "APPROVED"}
NON_US_AGENCY = re.compile(r"\b(EMA|European|EU|MAA|CHMP|Japan\w*|MHLW|PMDA|NMPA|China|Chinese|Canad\w*|TGA|"
                           r"MHRA|Swissmedic|ANVISA)\b")
# Pipeline tables and slide decks merge into giant "sentences"; a milestone in one is not reliably this drug's.
MAX_EVENT_SENTENCE = 500
DATE_WINDOW = 80               # a date must sit this close to the event phrase to date it
US_AGENCY = re.compile(r"\b(FDA|Food and Drug Administration|U\.S\.|United States|PDUFA|NDA|BLA|sNDA|sBLA)\b")
# With a non-US agency in the sentence, an application acronym is not evidence of a US filing ('The NDA in China
# was accepted by the NMPA and granted priority review'): the FDA itself must be named.
US_AGENCY_EXPLICIT = re.compile(r"\b(FDA|Food and Drug Administration|U\.S\.|United States|PDUFA)\b")
CONDITIONAL = re.compile(r"\b(if|once|unless|until|pending|subject to|whether|would be)\s*$", re.IGNORECASE)
NOT_APPROVED = re.compile(r"\b(not (yet )?(been )?(be )?approved|has not been|have not been|investigational (drug|"
                          r"medicine|product)|provisionally approved)\b", re.IGNORECASE)
IF_CLAUSE = re.compile(r"\b(if|once|upon|when|unless|assuming)\b", re.IGNORECASE)
NEGATED = re.compile(r"\b(did not|didn't|failed to|not meet|missed|was not|were not|no statistically)\b", re.IGNORECASE)

_P3 = r"(phase\s*3|phase\s*iii|phase\s*2/3|pivotal)"
_P2 = r"(phase\s*2|phase\s*ii|phase\s*2b|phase\s*2a|proof[- ]of[- ]concept)"
_POS = r"(positive (top-?line |topline )?(results|data)|met (its|the|their|all|both)? ?(\w+ )?(primary )?endpoints?|" \
       r"achieved (its|the|their) (\w+ )?primary endpoint|statistically significant)"
_APP = r"(NDA|BLA|sNDA|sBLA|MAA|new drug application|biologics license application)"

IND = re.compile(r"(?i:investigational new drug)|\bIND\b")
# The FDA accepting something other than an application ('accepted the notion of secondary hypogonadism as an
# appropriate population', 'accepted our proposed trial design').
ACCEPTED_OTHER = re.compile(r"accepted\s+(the|our|a|an|its|their)?\s*(proposed\s+)?(notion|concept|proposal|design|plan|"
                            r"protocol|endpoint|approach|population|trial|study|request|pathway|strategy)", re.IGNORECASE)
_PHASE = re.compile(r"\bphase\s*(1|2|3|i{1,3})[ab]?(?:\s*/\s*(2|3|ii|iii))?\b", re.IGNORECASE)
_POS_RE = re.compile(_POS + r"|did not (meet|achieve)|failed to (meet|achieve|demonstrate)|missed", re.IGNORECASE)


def _result_is_phase2(s: str, start: int, end: int) -> bool:
    """True if the result phrase in s[start:end] belongs to a Phase 2 (not 2/3) mention: the trial named just
    before it ('The Phase 2 portion met its primary endpoint, and the Phase 3 trial is ongoing'), else the one
    just after it ('Positive Results From Phase 2 Clinical Trial')."""
    r = _POS_RE.search(s, start, end)
    if not r:
        return False
    phases = list(_PHASE.finditer(s))
    before = [p for p in phases if p.end() <= r.start() and r.start() - p.end() <= 80]
    after = [p for p in phases if p.start() >= r.end()]
    subject = before[-1] if before else (after[0] if after else None)
    return bool(subject) and subject.group(1).lower() in ("2", "ii") and not subject.group(2)


# (event type, pattern, requires past tense, extra guard)
RULES: list[tuple[str, re.Pattern, bool]] = [
    ("PHASE3_NEGATIVE", re.compile(rf"{_P3}.*?(did not (meet|achieve)|failed to (meet|achieve|demonstrate)|missed)", re.IGNORECASE), True),
    ("PHASE3_POSITIVE", re.compile(rf"{_P3}.*?{_POS}|{_POS}.*?{_P3}", re.IGNORECASE), True),
    ("PHASE2_NEGATIVE", re.compile(rf"{_P2}.*?(did not (meet|achieve)|failed to (meet|achieve|demonstrate)|missed)", re.IGNORECASE), True),
    ("PHASE2_POSITIVE", re.compile(rf"{_P2}.*?{_POS}|{_POS}.*?{_P2}", re.IGNORECASE), True),
    ("NDA_SUBMITTED", re.compile(rf"\b(submitted|completed (the )?(rolling )?submission of|submission of)\b.{{0,60}}{_APP}", re.IGNORECASE), True),
    ("NDA_ACCEPTED", re.compile(rf"(FDA|Food and Drug Administration)\s+(has\s+)?accepted.{{0,80}}({_APP}|application|submission)|accepted for (filing|review)", re.IGNORECASE), True),
    ("PRIORITY_REVIEW", re.compile(r"(granted|with|received|under)\s+(a\s+)?priority review", re.IGNORECASE), True),
    # An assignment, not a comparison ('accelerates the standard review time', 'versus a standard review timeline').
    ("STANDARD_REVIEW", re.compile(r"\b(assign\w*|grant\w*|under(going)?|receiv\w*|given)\b.{0,60}?\bstandard review\b"
                                   r"(?!\s+(time|timeline|period|clock|cycle))", re.IGNORECASE), True),
    ("PDUFA_DATE", re.compile(rf"(PDUFA|target action date|goal date).{{0,60}}?({_MONTH_RE})\s+(\d{{1,2}},?\s+)?(\d{{4}})", re.IGNORECASE), False),
    ("PDUFA_EXTENDED", re.compile(r"(extend\w*|extension).{0,80}(PDUFA|goal date|action date|review (period|clock))|major amendment", re.IGNORECASE), True),
    ("ADCOM_NOT_PLANNED", re.compile(r"(not|no)\s+(currently\s+)?(plan\w*|intend\w*|expect\w*)\s+to\s+hold\s+an?\s+advisory committee|no advisory committee", re.IGNORECASE), False),
    # 'The Committee voted 21 to 5', 'AMDAC voted (7-6)', 'voted favorably 14 to 1', 'voted 11 to 8 against'.
    ("ADCOM_VOTE", re.compile(r"(?:advisory committee|committee|panel|\bODAC\b|\b[A-Z]{2,6}AC\b|adcom).{0,200}?\bvoted\b"
                              r"[^.0-9]{0,40}?\(?\s*(\d{1,2})\s*(?:to|-|–)\s*(\d{1,2})\s*\)?", re.IGNORECASE), True),
    ("ADCOM_SCHEDULED", re.compile(r"advisory committee (meeting|will (meet|convene))|(convene|schedul\w+) an? advisory committee", re.IGNORECASE), False),
    ("CRL", re.compile(r"complete response letter|\bCRL\b", re.IGNORECASE), True),
    ("APPROVED", re.compile(r"(FDA|Food and Drug Administration)\s+(has\s+)?(approved|granted (full |traditional |accelerated |regular )?approval)|approved (in|by) the (U\.S\.|FDA|United States)", re.IGNORECASE), True),
    ("BTD", re.compile(r"breakthrough therapy", re.IGNORECASE), True),
    ("FAST_TRACK", re.compile(r"fast track", re.IGNORECASE), True),
    ("ORPHAN", re.compile(r"orphan drug designation", re.IGNORECASE), True),
    ("SPA", re.compile(r"special protocol assessment", re.IGNORECASE), True),
    ("ACCELERATED_PATHWAY", re.compile(r"accelerated approval", re.IGNORECASE), False),
]
RULES_BY_TYPE = {etype: pattern for etype, pattern, _ in RULES}

# Primary-endpoint p-values reported with a readout ('met its primary endpoint (p<0.001)').
PVALUE = re.compile(r"\bp\s*(?:-?\s*values?)?\s*(?:(?:<|=|≤|&lt;|less than|of|was)\s*)+"
                    r"(\d(?:\.\d+)?\s*[x×]\s*10\s*[-−–]\s*\d+|\d(?:\.\d+)?e-\d+|0?\.\d+|1\.0)", re.IGNORECASE)
PVALUE_WINDOW = 8              # sentences after the readout sentence searched for its p-value
_NOT_PRIMARY = re.compile(r"secondary|post.?hoc|sub.?group|nominal|exploratory|pooled|meta.?analys", re.IGNORECASE)


def _pvalue_number(raw: str) -> float | None:
    s = re.sub(r"\s+", "", raw).replace("×", "x").replace("−", "-").replace("–", "-")
    try:
        if "x10" in s:
            mant, exp = s.split("x10")
            return float(mant) * 10 ** float(exp)
        return float(s)
    except ValueError:
        return None


def readout_pvalue(sentences: list[str], i: int, phase: str) -> str | None:
    """Worst (largest) primary-endpoint p-value reported for the readout in sentence `i`, searched in that sentence
    and the next PVALUE_WINDOW: a later sentence must say 'primary'. Sentences about secondary/subgroup analyses or
    another phase, and the 'About ...' boilerplate, are skipped. `phase` is '2' or '3'."""
    for k, s in enumerate(sentences[i:i + PVALUE_WINDOW + 1]):
        if k and re.match(r"\s*About\b", s):
            break
        found = [p for p in (_pvalue_number(m.group(1)) for m in PVALUE.finditer(s)) if p is not None and 0 < p <= 1]
        if not found or (k and not re.search(r"primary", s, re.IGNORECASE)) or _NOT_PRIMARY.search(s):
            continue
        phases = {(m.group(2) or m.group(1)).lower() for m in _PHASE.finditer(s)}
        phases = {"3" if p in ("3", "iii") else "2" if p in ("2", "ii") else "1" for p in phases}
        if (phases and phase not in phases) or (k and phases - {phase}):   # a later sentence may name only this phase
            continue
        return f"{max(found):g}"
    return None


@dataclass
class ParsedEvent:
    event_type: str
    event_date: str            # best estimate of when it happened
    announced_date: str        # filing date of the document
    date_basis: str            # 'explicit' | 'month' | 'quarter' | 'filing_date'
    quote: str
    source_id: str
    value: str | None = None   # e.g. PDUFA date, AdCom vote "12-2"
    indication_match: str = "context"   # 'sentence' | 'context'
    name_match: str = "sentence"        # drug named in the event sentence, or only in the preceding ones

    def to_dict(self) -> dict:
        return asdict(self)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z•\"“])|\s*•\s*|\s+—\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 15]


def _mention_spans(sentence: str, names: list[str]) -> list[tuple[int, int]]:
    s = sentence.lower()
    return [m.span() for n in names for m in re.finditer(rf"(?<![a-z0-9]){re.escape(n.lower())}(?![a-z0-9])", s)]


def _mentions(sentence: str, names: list[str]) -> bool:
    return bool(_mention_spans(sentence, names))


def _near(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(max(a - end, start - b, 0) <= MAX_NAME_DISTANCE for a, b in spans)


def _keyword_coverage(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return sum(any(fuzz.ratio(k, w) >= 85 for w in words) for k in keywords) / len(keywords)


def _attributed_elsewhere(sentence: str, names: list[str], keywords: list[str]) -> bool:
    """True if the sentence ties the drug to a different indication, e.g. 'VX-548 in diabetic neuropathy'."""
    for n in names:
        for m in re.finditer(rf"{re.escape(n)}\)?\s+(?:in|for(?: the treatment of)?)\s+((?:[\w-]+\s*){{1,6}})",
                             sentence, re.IGNORECASE):
            target = m.group(1)
            if _keyword_coverage(target, keywords) == 0 and not re.match(r"(the|a|an|this|these|both|all)\b", target, re.IGNORECASE):
                return True
    return False


_GENERIC_TARGETS = {"the", "a", "an", "this", "these", "both", "all", "adults", "adult", "patients", "people",
                    "u", "us", "united", "states", "fda", "review", "development", "use", "approval", "our"}
_TARGET = re.compile(r"\b(?:in|for)\s+(?:the\s+treatment\s+of\s+)?(?:patients\s+with\s+)?([A-Za-z][\w-]*(?:\s+[a-z][\w-]*){0,3})")
_SAFE_HARBOR = re.compile(r"(special note regarding forward-looking|forward-looking statements|"
                          r"statements in this press release)", re.IGNORECASE)


def _names_other_target(fragment: str, keywords: list[str]) -> bool:
    """True if `fragment` says 'in/for <X>' where X shares nothing with the indication (e.g. 'in DPN')."""
    for m in _TARGET.finditer(fragment):
        target = m.group(1)
        first = target.split()[0].lower().strip(".")
        if first in _GENERIC_TARGETS:
            continue
        if _keyword_coverage(target, keywords) == 0:
            return True
    return False


def _period_end(year: int, last_month: int) -> date:
    return date(year, last_month, calendar.monthrange(year, last_month)[1])


def _event_date(sentence: str, filed: date) -> tuple[date, str]:
    """When the event became public. A month or quarter ('In March', 'in the first quarter of 2015') gives the
    *last* day of that period, capped at the filing date: the latest date consistent with the text, so a level
    cut-off built on it never precedes the announcement."""
    if m := DATE_FULL.search(sentence):
        d = date(int(m.group(3)), MONTHS.index(m.group(1).lower()) + 1, int(m.group(2)))
        if d <= filed:
            return d, "explicit"
    if m := DATE_MONTH_YEAR.search(sentence):
        month = MONTHS.index(m.group(1).lower()) + 1
        year = int(m.group(2)) if m.group(2) else (filed.year if month <= filed.month else filed.year - 1)
        if date(year, month, 1) <= filed:
            return min(_period_end(year, month), filed), "month"
    if m := DATE_QUARTER.search(sentence):
        q = ["first", "second", "third", "fourth"].index(m.group(1).lower())
        year = int(m.group(2))
        if date(year, 3 * q + 1, 1) <= filed:
            return min(_period_end(year, 3 * q + 3), filed), "quarter"
    return filed, "filing_date"


def body_sentences(text: str) -> list[str]:
    if m := _SAFE_HARBOR.search(text):      # boilerplate lists every possible risk; never evidence of an event
        text = text[:m.start()]
    return split_sentences(text)


def parse_document(text: str, filed: date, source_id: str, names: list[str], indication: str,
                   indication_keywords: list[str], sentences: list[str] | None = None) -> list[ParsedEvent]:
    """`sentences` lets a bulk caller split a document once and parse it for many programmes."""
    if sentences is None:
        sentences = body_sentences(text)
    events: list[ParsedEvent] = []
    for i, s in enumerate(sentences):
        window = " ".join(sentences[max(0, i - 3):i + 1])
        spans = _mention_spans(s, names)
        in_sentence = bool(spans)
        if not in_sentence and (len(s) > MAX_CONTEXT_SENTENCE
                                or not _mentions(" ".join(sentences[max(0, i - 2):i]), names)):
            continue
        if in_sentence and _attributed_elsewhere(s, names, indication_keywords):
            continue
        sentence_cov = _keyword_coverage(s, indication_keywords)
        if sentence_cov < 0.5 and _keyword_coverage(window, indication_keywords) < 0.5:
            continue
        for etype, pattern, past_only in RULES:
            m = pattern.search(s)
            if not m:
                continue
            if past_only and FUTURE.search(s[:m.end()]) and etype not in ("BTD", "FAST_TRACK", "ORPHAN"):
                continue
            if in_sentence and not _near(spans, m.start(), m.end()):
                continue
            if etype in REGULATORY and NON_US_AGENCY.search(s) and not US_AGENCY_EXPLICIT.search(s):
                continue
            if (etype in REGULATORY or etype.startswith("PHASE3")) and len(s) > MAX_EVENT_SENTENCE:
                continue
            if etype in ("NDA_SUBMITTED", "NDA_ACCEPTED") and IND.search(s):
                continue
            if etype == "NDA_ACCEPTED" and ACCEPTED_OTHER.search(s):
                continue
            # 'The Phase 2 portion met its primary endpoint, and the Phase 3 trial is ongoing'; 'approved ... based on
            # positive data from the phase 3 ARIEL3 trial' (an approval, not a readout).
            if etype.startswith("PHASE3") and (_result_is_phase2(s, m.start(), m.end())
                                               or RULES_BY_TYPE["APPROVED"].search(s)):
                continue
            if sentence_cov < 0.5 and _names_other_target(s[m.start():], indication_keywords):
                continue
            if etype == "ACCELERATED_PATHWAY" and (not in_sentence or FUTURE.search(s)):
                continue
            # An approval must name the drug in the same sentence ('X is a novel antibiotic approved by the FDA'
            # in a paragraph about another drug) and not be conditional ('If approved by the FDA, ...').
            if etype == "APPROVED" and (not in_sentence or CONDITIONAL.search(s[max(0, m.start() - 20):m.start()])
                                        or NOT_APPROVED.search(s) or IF_CLAUSE.search(s[:m.start()])):
                continue
            if etype.endswith("POSITIVE") and NEGATED.search(s):
                continue
            if etype in ("BTD", "FAST_TRACK", "ORPHAN") and not re.search(r"\b(granted|received|has|have|awarded)\b", s, re.IGNORECASE):
                continue
            value = None
            if etype == "PDUFA_DATE":
                dm = DATE_FULL.search(s[m.start():]) or None
                value = (date(int(dm.group(3)), MONTHS.index(dm.group(1).lower()) + 1, int(dm.group(2))).isoformat()
                         if dm else None)
                if value is None:
                    continue
            if etype == "ADCOM_VOTE":
                yes, no = m.group(1), m.group(2)
                if re.match(r"[\s,)]*(?:votes?\s+)?against", s[m.end():m.end() + 30], re.IGNORECASE) or \
                        re.search(r"against", s[m.start():m.start(1)], re.IGNORECASE):
                    yes, no = no, yes                       # 'voted 11 to 8 against' = 8 for, 11 against
                value = f"{yes}-{no}"
            if etype.startswith(("PHASE2", "PHASE3")):
                value = readout_pvalue(sentences, i, etype[5])
            d, basis = _event_date(s[max(0, m.start() - DATE_WINDOW):m.end() + DATE_WINDOW], filed)
            events.append(ParsedEvent(etype, d.isoformat(), filed.isoformat(), basis, s[:600], source_id, value,
                                      "sentence" if sentence_cov >= 0.5 else "context",
                                      "sentence" if in_sentence else "context"))
            if etype in ("PHASE3_NEGATIVE", "PHASE2_NEGATIVE"):
                break   # a negative readout sentence must not also count as positive
    return events


def parse_filing_date(value: str) -> date:
    d = parse_partial(value)
    if d is None:
        raise ValueError(f"bad filing date {value!r}")
    return d
