from datetime import date

from fdaengine.resolve.programme import indication_keywords
from fdaengine.timeline.press_release_parser import parse_document

NAMES = ["suzetrigine", "VX-548"]
IND = "moderate to severe acute pain"
KW = indication_keywords(IND)


def parse(text, filed=date(2024, 8, 1)):
    return {e.event_type: e for e in parse_document(text, filed, "doc", NAMES, IND, KW)}


def test_acceptance_priority_and_pdufa_in_one_sentence():
    ev = parse("FDA accepted NDA for suzetrigine (VX-548) for moderate-to-severe acute pain with Priority Review "
               "and PDUFA target action date of January 30, 2025.")
    assert {"NDA_ACCEPTED", "PRIORITY_REVIEW", "PDUFA_DATE"} <= set(ev)
    assert ev["PDUFA_DATE"].value == "2025-01-30"


def test_explicit_and_month_dates():
    ev = parse("On January 30, 2025, the FDA approved suzetrigine for adults with moderate-to-severe acute pain.",
               filed=date(2025, 2, 10))
    assert ev["APPROVED"].event_date == "2025-01-30" and ev["APPROVED"].date_basis == "explicit"
    ev = parse("In March, Vertex reported positive data from two Phase 2 acute pain studies with VX-548.",
               filed=date(2022, 5, 5))
    assert ev["PHASE2_POSITIVE"].event_date == "2022-03-31"      # end of the month: never before it was public


def test_other_indication_is_rejected():
    assert parse("In December 2023, Vertex shared positive Phase 2 results with VX-548 in diabetic peripheral "
                 "neuropathy.") == {}
    assert "BTD" not in parse("Suzetrigine is being developed for acute pain. "
                              "The FDA has granted suzetrigine Breakthrough Therapy Designation in DPN.")


def test_future_tense_and_negation():
    assert "NDA_SUBMITTED" not in parse("Vertex plans to submit an NDA for VX-548 in acute pain by mid-2024.")
    ev = parse("The Phase 3 trial of VX-548 in acute pain did not meet its primary endpoint.")
    assert "PHASE3_NEGATIVE" in ev and "PHASE3_POSITIVE" not in ev


def test_approval_must_be_factual_and_name_the_drug():
    assert "APPROVED" not in parse("If approved by the FDA, suzetrigine will become the first new class of medicine "
                                   "for acute pain in decades.", filed=date(2024, 12, 1))
    assert "APPROVED" not in parse("Vertex is developing suzetrigine for acute pain. JOURNAVX-X is a non-opioid "
                                   "approved by the FDA for acute pain.", filed=date(2024, 12, 1))
    assert "APPROVED" not in parse("If suzetrigine is approved by the FDA for acute pain, Vertex will receive a "
                                   "milestone.", filed=date(2024, 12, 1))
    assert "APPROVED" not in parse("Suzetrigine is an investigational medicine that has not been approved by the "
                                   "FDA for acute pain.", filed=date(2024, 12, 1))
    assert "APPROVED" in parse("The FDA has approved suzetrigine for moderate-to-severe acute pain.",
                               filed=date(2025, 2, 1))


def test_events_in_multi_drug_documents_need_the_drug_nearby():
    # Another drug's PDUFA date in a release that mentions suzetrigine earlier in the paragraph.
    other = ("Vertex continues to enrol the suzetrigine acute pain programme. NDA for VX-121 for the treatment of "
             "cystic fibrosis in patients aged 6 and older across a broad range of genotypes, accepted with a "
             "Prescription Drug User Fee Act (PDUFA) date of January 2, 2025, following a submission package that "
             "included three pivotal studies and an extensive open-label extension safety database.")
    assert "PDUFA_DATE" not in parse(other)
    table = ("Select 2024 Milestones Programs Milestones Pain - Report suzetrigine Phase 3 acute pain results "
             + "Cardiology - Submit sNDA for an unrelated product in heart failure " * 6
             + "- FDA accepted for review the NDA for VX-880 with PDUFA date of March 3, 2025")
    assert not {"NDA_ACCEPTED", "PDUFA_DATE"} & set(parse(table))


def test_non_us_regulatory_events_are_ignored():
    assert "NDA_ACCEPTED" not in parse("The European Medicines Agency (EMA) accepted for review the marketing "
                                       "authorisation application for suzetrigine in acute pain.")
    assert "NDA_ACCEPTED" in parse("The FDA accepted for review the NDA for suzetrigine in acute pain; the EMA "
                                   "validated the MAA.")


def test_safe_harbor_boilerplate_ignored():
    text = ("VX-548 continues in acute pain. Special Note Regarding Forward-Looking Statements: the FDA may not "
            "grant accelerated approval to VX-548 for acute pain and a complete response letter could be issued.")
    assert parse(text) == {}


# ---- gold-set regressions (2026-10-30) ----
def test_ind_acceptance_phase2_portion_and_canada_are_not_events():
    assert "NDA_ACCEPTED" not in parse("In the second quarter, the FDA accepted our investigational new drug "
                                       "application (IND) for suzetrigine in acute pain.")
    assert "PHASE3_POSITIVE" not in parse("The Phase 2 portion of the suzetrigine acute pain study met its primary "
                                          "endpoint, and the Phase 3 trial is ongoing.")
    assert "PRIORITY_REVIEW" not in parse("The Canadian application for suzetrigine in acute pain was granted "
                                          "priority review status.")


def test_event_date_comes_from_near_the_event_phrase():
    # Arazlo: a quarter phrase about something else in the same sentence dated the NDA acceptance 5 months early.
    ev = parse("Product revenue grew in the third quarter of 2018 across all of our dermatology and eye-care brands "
               "worldwide; separately, the FDA accepted the NDA for suzetrigine for moderate-to-severe acute pain.",
               filed=date(2019, 5, 1))
    assert ev["NDA_ACCEPTED"].date_basis == "filing_date"
    ev = parse("In March 2019, the FDA accepted the NDA for suzetrigine for acute pain.", filed=date(2019, 5, 1))
    assert ev["NDA_ACCEPTED"].event_date == "2019-03-31"


# ---- L3 regressions (2026-11-01) ----
def test_non_us_filing_with_application_acronym_is_ignored():
    assert "PRIORITY_REVIEW" not in parse("The new drug application (NDA) in China for suzetrigine in acute pain has "
                                          "been accepted by the China National Medical Products Administration "
                                          "(NMPA) and granted priority review.")
    assert "PRIORITY_REVIEW" in parse("The FDA granted priority review to the NDA for suzetrigine in acute pain; a "
                                      "filing in China is planned.")


def test_capitalised_ind_acceptance_is_not_a_filing():
    assert "NDA_ACCEPTED" not in parse("The U.S. Food and Drug Administration has accepted a new Investigational New "
                                       "Drug application for the Phase 2 trial of suzetrigine in acute pain.")


def test_standard_review_must_be_assigned():
    assert "STANDARD_REVIEW" in parse("The FDA assigned the suzetrigine NDA for acute pain a standard review.")
    assert "STANDARD_REVIEW" not in parse("Priority review for suzetrigine in acute pain would shorten the review "
                                          "versus a standard review timeline of 12 months.")
    assert "STANDARD_REVIEW" not in parse("Fast Track status for suzetrigine in acute pain accelerates the standard "
                                          "review time from 10 to six months.")


def test_fda_accepting_a_concept_is_not_a_filing():
    assert "NDA_ACCEPTED" not in parse("The FDA accepted the notion of moderate to severe acute pain after surgery as an "
                                       "appropriate population for the suzetrigine application.")
    assert "NDA_ACCEPTED" in parse("The FDA accepted the NDA for suzetrigine in moderate to severe acute pain.")
