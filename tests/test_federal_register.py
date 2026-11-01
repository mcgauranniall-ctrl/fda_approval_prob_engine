from datetime import date

from fdaengine.sources import federal_register as fr

NOTICE = """Oncologic Drugs Advisory Committee; Notice of Meeting
DATES: The meeting will be held on March 18, 2015, from 8 a.m. to 5 p.m.
Agenda: The committee will discuss new drug application (NDA)
022225, drugx injection, submitted by Acme Inc., for the proposed indication of relapsed multiple myeloma.
Procedure: Interested persons may present data.
The meeting will be held on March 19, 2015.
Agenda: The committee will discuss biologics license application (BLA) 125514, drugy, for melanoma.
Procedure: ..."""


def test_agenda_sections_dates_and_applications():
    segs = fr.agenda_segments(NOTICE)
    assert len(segs) == 2 and "drugx" in segs[0][1] and "drugy" not in segs[0][1]
    dates = fr.meeting_dates(NOTICE)
    assert fr.meeting_date_near(dates, segs[0][0]) == date(2015, 3, 18)
    assert fr.meeting_date_near(dates, segs[1][0]) == date(2015, 3, 19)
    assert fr.applications(segs[0][1]) == {"022225"} and fr.applications(segs[1][1]) == {"125514"}


def test_only_drugs_named_with_an_application_are_meeting_subjects():
    seg = ("Agenda: The committee will discuss biologics license application (BLA) 761028 for ABP 215, a proposed "
           "biosimilar to Genentech's AVASTIN (bevacizumab). The committee will also discuss long-term opioid use.")
    (sent, apps), = fr.application_sentences(seg)
    assert apps == {"761028"} and "ABP 215" in sent and "bevacizumab" not in sent
