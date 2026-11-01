from fdaengine.dataset import labels as lab
from fdaengine.resolve.programme import indication_keywords

AS_OF = "2026-10-29"
JOURNAVX_LABEL = ("JOURNAVX is a sodium channel blocker indicated for the treatment of moderate to severe acute pain "
                  "in adults.")
ACUTE = indication_keywords("Acute Pain")
DPN = indication_keywords("Diabetic Peripheral Neuropathic Pain")


def journavx(keywords, holder=True):
    return lab.Action("2025-01-30", "219209", "ORIG", "TYPE 1", holder,
                      lab.indication_coverage(keywords, JOURNAVX_LABEL))


def test_label_coverage_uses_indication_wording():
    assert lab.indication_coverage(ACUTE, JOURNAVX_LABEL) == 1.0
    assert lab.indication_coverage(DPN, JOURNAVX_LABEL) < lab.LABEL_COVERAGE
    assert lab.indication_coverage(indication_keywords("Breast Neoplasms"),
                                   "indicated for HER2-positive metastatic breast cancer") >= lab.LABEL_COVERAGE


def test_trial_condition_wording_also_counts():
    label = "NUZYRA is indicated for community-acquired bacterial pneumonia and acute bacterial skin and skin " \
            "structure infections (ABSSSI)."
    mesh = indication_keywords("Soft Tissue Infections")
    trial = indication_keywords("Acute Bacterial Skin and Skin Structure Infections")
    assert lab.indication_coverage(mesh, label) < lab.LABEL_COVERAGE
    assert lab.best_coverage(lab.keyword_sets(mesh, [trial]), label) >= lab.LABEL_COVERAGE
    # A bare generic condition ('Pain') must not let any pain label cover a DPN programme.
    sets = lab.keyword_sets(DPN, [["pain"], ["healthy", "volunteers"]])
    assert sets == [DPN] and lab.best_coverage(sets, JOURNAVX_LABEL) < lab.LABEL_COVERAGE


def test_cber_products_are_recognised():
    assert all(map(lab.is_cber_product, ["RECOMBINANT VARICELLA ZOSTER VIRUS GLYCOPROTEIN E ANTIGEN",
                                         "tisagenlecleucel", "talimogene laherparepvec", "HUMAN IMMUNOGLOBULIN G",
                                         "VONICOG ALFA", "C1 ESTERASE INHIBITOR HUMAN"]))
    assert not any(map(lab.is_cber_product, ["suzetrigine", "PEMBROLIZUMAB", "SEMAGLUTIDE", "cellulose"]))


def test_suzetrigine_first_approval_goes_to_acute_pain_only():
    acute = lab.Facts("PG1", "2022-10-03", False, actions=[journavx(ACUTE)], press_approvals=["2025-01-30"])
    out = lab.programme_label(acute, AS_OF)
    assert (out["y_prog"], out["event"], out["approval_date"]) == (1, 1, "2025-01-30")
    assert (out["label_source"], out["label_confidence"]) == ("edgar+drugsfda", "high")

    # Same approval, DPN programme: the label does not cover DPN, and the press release was about acute pain.
    dpn = lab.Facts("PG2", "2024-10-01", False, n_rival_programmes=1, actions=[journavx(DPN)],
                    p3_statuses=["RECRUITING"], p3_end_dates=["2026-06-01"])
    out = lab.programme_label(dpn, AS_OF)
    assert out["approval_date"] is None and out["y_prog"] is None and out["event"] == 0     # censored, window open


def test_first_approval_without_press_release_uses_label():
    f = lab.Facts("PG1", "2019-01-01", False, actions=[journavx(ACUTE, holder=False)])
    out = lab.programme_label(f, AS_OF)
    assert out["y_prog"] == 1 and (out["label_source"], out["label_confidence"]) == ("drugsfda_orig+label", "medium")


def test_unlabelled_first_approval_needs_a_sole_programme():
    act = lab.Action("2012-05-01", "020001", "ORIG", "TYPE 1", True, None)
    sole = lab.programme_label(lab.Facts("PG1", "2008-01-01", False, 0, [act]), AS_OF)
    assert sole["y_prog"] == 1 and sole["label_source"] == "drugsfda_orig_sole_programme"
    shared = lab.programme_label(lab.Facts("PG1", "2008-01-01", False, 2, [act]), AS_OF)
    assert shared["event"] is None and shared["y_prog"] is None
    assert shared["label_source"] == "ambiguous_unlabelled_approval"


def test_lifecycle_needs_holder_and_a_supplement_after_the_phase3():
    label = "indicated for plaque psoriasis and psoriatic arthritis"
    kw = indication_keywords("Arthritis, Psoriatic")
    cov = lab.indication_coverage(kw, label)
    early = lab.Action("2014-03-01", "125000", "SUPPL", "EFFICACY", True, cov)     # before this Phase 3 completed
    late = lab.Action("2018-06-01", "125000", "SUPPL", "EFFICACY", True, cov)
    other = lab.Action("2017-01-01", "761000", "ORIG", "BIOSIMILAR", False, cov)   # someone else's application
    f = lab.Facts("PG1", "2015-01-01", True, actions=[early, other, late], p3_end_dates=["2016-12-01"])
    out = lab.programme_label(f, AS_OF)
    assert out["approval_date"] == "2018-06-01" and out["label_confidence"] == "low"
    f.actions = [early, other]
    assert lab.programme_label(f, AS_OF)["y_prog"] == 0                             # window closed, nothing after


def test_failures_and_window():
    term = lab.Facts("PG1", "2023-01-01", False, p3_statuses=["TERMINATED", "WITHDRAWN"],
                     p3_end_dates=["2023-09-01", "2023-05-01"])
    out = lab.programme_label(term, AS_OF)
    assert (out["event"], out["y_prog"], out["event_date"], out["failure_basis"]) == \
        (2, 0, "2023-09-01", "all_p3_terminated")
    late = lab.Facts("PG2", "2010-01-01", False,
                     actions=[lab.Action("2018-03-01", "1", "ORIG", "TYPE 1", True, 1.0)])
    out = lab.programme_label(late, AS_OF)
    assert (out["y_prog"], out["failure_basis"], out["event_date"]) == (0, "approved_after_window", "2017-01-01")
    open_ = lab.Facts("PG3", "2024-01-01", False, p3_statuses=["RECRUITING"])
    assert lab.programme_label(open_, AS_OF)["event"] == 0


def test_press_only_approval_is_not_a_label():
    f = lab.Facts("PG1", "2018-01-01", False, press_approvals=["2021-06-01"], p3_statuses=["COMPLETED"],
                  p3_end_dates=["2019-01-01"])
    out = lab.programme_label(f, AS_OF)
    assert out["label_source"] == "edgar_only_unverified" and out["event"] is None and out["y_prog"] is None


def test_press_approval_before_programme_is_ignored():
    f = lab.Facts("PG1", "2012-05-16", False, press_approvals=["2011-07-14"], p3_statuses=["COMPLETED"],
                  p3_end_dates=["2014-01-01"])
    out = lab.programme_label(f, AS_OF)
    assert out["approval_date"] is None and out["y_prog"] == 0


def test_next_action():
    assert lab.next_action("2024-08-02", "2025-01-30", [], AS_OF)["y_next"] == 1
    crl_first = lab.next_action("2019-03-01", "2020-06-01", ["2019-12-20"], AS_OF)
    assert (crl_first["y_next"], crl_first["next_action_date"]) == (0, "2019-12-20")
    assert lab.next_action("2026-03-01", None, [], AS_OF)["next_action_status"] == "pending"
    assert lab.next_action("2015-03-01", None, [], AS_OF)["next_action_status"] == "no_action_found"


def test_level_target():
    # L1-L2 keep Y_PROG (D1)
    assert lab.level_target("L2", "2016-01-01", None, 0, 2, 900, AS_OF) == \
        {"y_level": 0, "event_level": 2, "t_level_days": 900}
    # filed in year 6 of the window, approved after it: a D1 failure, a Y_LEVEL approval
    assert lab.level_target("L3", "2021-03-01", "2022-03-18", 0, 2, -200, AS_OF)["y_level"] == 1
    # no approval within 3 years of the cut-off
    assert lab.level_target("L5", "2015-01-01", "2019-06-01", 0, 2, 100, AS_OF)["event_level"] == 2
    # horizon still open: censored
    assert lab.level_target("L3", "2025-01-01", None, None, 0, None, AS_OF) == \
        {"y_level": None, "event_level": 0, "t_level_days": 666}
    # untrusted programme label stays unlabelled
    assert lab.level_target("L3", "2015-01-01", None, None, None, None, AS_OF)["y_level"] is None


# ---- gold-set regressions (2026-10-30) ----
def test_negated_label_text_and_unspecific_words_do_not_cover():
    lyrica_cr = ("LYRICA CR is indicated for the management of: Neuropathic pain associated with diabetic peripheral "
                 "neuropathy. Efficacy of LYRICA CR has not been established for the management of fibromyalgia.")
    assert lab.best_coverage([indication_keywords("Fibromyalgia")], lyrica_cr) < lab.LABEL_COVERAGE
    ilaris = "ILARIS is indicated for Periodic Fever Syndromes in adults and children, and Still's disease (non-small)."
    assert lab.best_coverage([indication_keywords("Carcinoma, Non-Small-Cell Lung")], ilaris) < lab.LABEL_COVERAGE


def test_approval_before_this_programmes_phase3_belongs_to_another_programme():
    absssi = lab.Action("2017-06-19", "208610", "ORIG", "TYPE 1", True, 1.0)
    cabp = lab.Action("2019-10-24", "208610", "SUPPL", "EFFICACY", True, 1.0)
    f = lab.Facts("PG1", "2016-12-14", False, n_rival_programmes=1, actions=[absssi, cabp],
                  p3_end_dates=["2018-07-31"], rival_p3_ends=["2016-03-31"])
    assert lab.programme_label(f, AS_OF)["approval_date"] == "2019-10-24"


def test_early_approval_counts_when_no_rival_had_finished_phase3():
    # Krazati (accelerated, on Phase 2 data) came before KRYSTAL-12's completion date; the rectal programme's Phase 3
    # had not finished either, so the approval is the NSCLC programme's.
    krazati = lab.Action("2022-12-12", "216340", "ORIG", "TYPE 1", True, 1.0)
    f = lab.Facts("PG1", "2020-12-02", False, n_rival_programmes=1, actions=[krazati],
                  p3_end_dates=["2024-05-01"], rival_p3_ends=["2025-09-30"])
    assert lab.programme_label(f, AS_OF)["approval_date"] == "2022-12-12"


def test_earliest_candidate_wins_and_unlabelled_original_is_not_skipped():
    orig = lab.Action("2018-05-24", "761079", "ORIG", "TYPE 1", True, 1.0)
    later = lab.Action("2020-10-06", "761079", "SUPPL", "EFFICACY", True, 1.0)
    f = lab.Facts("PG1", "2013-05-01", False, actions=[orig, later], press_approvals=["2020-10-07"])
    assert lab.programme_label(f, AS_OF)["approval_date"] == "2018-05-24"
    orbactiv = lab.Action("2014-08-06", "206334", "ORIG", "TYPE 1", True, None)
    kimyrsa = lab.Action("2021-03-12", "214155", "ORIG", "TYPE 5", True, 1.0)
    f = lab.Facts("PG2", "2010-12-01", False, n_rival_programmes=1, actions=[orbactiv, kimyrsa],
                  p3_end_dates=["2012-10-31"])
    assert lab.programme_label(f, AS_OF)["approval_date"] == "2014-08-06"


def test_sole_programme_rule_needs_an_unlabelled_approval():
    chirhostim = lab.Action("2004-04-09", "021256", "ORIG", "TYPE 1", True, 0.0)   # label: pancreatic diagnostics
    f = lab.Facts("PG1", "2002-04-01", False, actions=[chirhostim])
    assert lab.programme_label(f, AS_OF)["y_prog"] == 0
