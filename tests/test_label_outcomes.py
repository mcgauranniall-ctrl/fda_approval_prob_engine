import numpy as np

from fdaengine.dataset import label_outcomes as lo
from fdaengine.model import label_model as lbm

PAD = " Silver Spring, MD 20993 www.fda.gov" * 20


def test_accelerated_list_rows_have_application_supplement_and_approval_date():
    text = ("BLA-761183\nSupplement 10\nTZIELD\nTEPLIZUMAB-MZWV\nProvention Bio, Inc.\n7/21/2025\n6/12/2026\n10.7\n"
            "TO DELAY ...\nNot Yet Converted\nNDA 050698\nBIAXIN\nCLARITHROMYCIN\nABBVIE\n11/2/1992\n12/23/1993\n13.7\n"
            "FOR THE TREATMENT ...\nConverted\n5/24/2002\n")
    rows = lo.parse_accelerated_list(text)
    assert [(r["ApplNo"], r["supplement"], r["aa_date"], r["aa_status"]) for r in rows] == [
        ("761183", "10", "2026-06-12", "Not Yet Converted"), ("050698", None, "1993-12-23", "Converted")]


def test_letter_accelerated_ignores_conversion_letters():
    grant = "This application is approved under the provisions of accelerated approval regulations (21 CFR 314.510)."
    convert = ("This supplement provides for conversion of the accelerated approval to traditional approval and "
               "fulfills the postmarketing requirement under 21 CFR 314.510.")
    assert lo.letter_accelerated(grant + PAD) is True
    assert lo.letter_accelerated(convert + PAD) is False
    assert lo.letter_accelerated("We have approved your application." + PAD) is False
    assert lo.letter_accelerated("") is None


def test_letter_rems_and_elements_to_assure_safe_use():
    rems = ("In accordance with section 505-1 of FDCA, we have determined that a REMS is necessary for Blenrep. "
            "Elements to assure safe use: prescribers are specially certified. Your proposed REMS is approved.")
    medguide = "We have determined that a REMS is necessary. Your proposed REMS consists of a Medication Guide."
    none = "We have completed our review of this application. It is approved."
    assert lo.letter_rems(rems + PAD) == (True, True)
    assert lo.letter_rems(medguide + PAD) == (True, False)
    assert lo.letter_rems(none + PAD) == (False, False)
    assert lo.letter_rems("We have determined that a REMS is not necessary for X." + PAD) == (False, False)
    declined = ("We acknowledge receipt of your submission of a proposed risk evaluation and mitigation strategy. "
                "We have determined that, at this time, a REMS is not necessary for Mepsevii.")
    assert lo.letter_rems(declined + PAD) == (False, False)


def test_boxed_warning_from_the_label_as_approved():
    plr_bw = ("HIGHLIGHTS OF PRESCRIBING INFORMATION WARNING: HEPATOTOXICITY See full prescribing information for "
              "complete boxed warning. INDICATIONS AND USAGE X is indicated for ..." + PAD)
    plr = "HIGHLIGHTS OF PRESCRIBING INFORMATION INDICATIONS AND USAGE X is ... WARNINGS AND PRECAUTIONS ..." + PAD
    old_bw = "DRUGNAME WARNING Fatal hepatic failure has occurred. DESCRIPTION ..." + PAD
    old = "DRUGNAME DESCRIPTION ... CLINICAL PHARMACOLOGY ... WARNINGS ... PRECAUTIONS" + PAD
    assert lo.label_boxed_warning(plr_bw) is True
    assert lo.label_boxed_warning(plr) is False
    assert lo.label_boxed_warning(old_bw) is True
    assert lo.label_boxed_warning(old) is False
    assert lo.label_boxed_warning("") is None


def test_label_indications_section():
    t = ("HIGHLIGHTS ... INDICATIONS AND USAGE X is a kinase inhibitor indicated for adults with relapsed or "
         "refractory mantle cell lymphoma. (1) DOSAGE AND ADMINISTRATION 200 mg daily")
    assert lo.label_indications(t).strip().startswith("X is a kinase inhibitor")
    assert "200 mg" not in lo.label_indications(t)


def test_narrower_label_needs_a_restriction_the_trials_lack():
    ind = "X is indicated for adult patients with moderate to severe plaque psoriasis who have failed prior therapy."
    broad_trial = [{"text": "A study of X in adults with plaque psoriasis. Inclusion: plaque psoriasis for 6 months; "
                            "mild, moderate or severe disease" + " detail" * 20, "min_age": 18.0}]
    pretreated_trial = [{"text": "A study of X in psoriasis patients who failed prior therapy with a biologic. "
                                 "Inclusion: moderate to severe disease" + " detail" * 20, "min_age": 18.0}]
    nar, why = lo.narrower(ind, broad_trial)
    assert nar is True and {w.category for w in why} == {"prior_therapy", "severity"}
    assert lo.narrower(ind, pretreated_trial) == (False, [])
    assert lo.narrower(ind, [])[0] is None


def test_narrower_label_by_age_and_diagnostic_subgroup():
    ind = ("X is indicated for adults with metastatic NSCLC whose tumors have a KRAS G12C mutation, as determined by "
           "an FDA-approved test.")
    trial = [{"text": "Phase 3 study in patients 12 years and older with advanced solid cancer" + " detail" * 20,
              "min_age": 12.0}]
    nar, why = lo.narrower(ind, trial)
    assert nar is True and {w.category for w in why} == {"diagnostic_subgroup", "age"}
    assert lo.label_min_age("for pediatric patients 2 years of age and older") == 2.0
    assert lo.trial_min_age("6 Months") == 0.5


def test_label_model_learns_a_strong_effect_and_shrinks_to_the_base_rate():
    rng = np.random.default_rng(1)
    n = 600
    feats = [{"therapeutic_area": "oncology" if rng.random() < 0.3 else "neurology",
              "accelerated_pathway": bool(rng.random() < 0.2)} for _ in range(n)]
    levels = ["L3"] * n
    raw = lbm.raw_matrix(levels, feats)
    p = np.where([f["accelerated_pathway"] for f in feats], 0.8, 0.05)
    y = (rng.random(n) < p).astype(float)
    y[:20] = np.nan                                                     # unknown outcomes are skipped
    m = lbm.LabelModel.fit(levels, raw, {"accelerated": y}, np.arange(n), n_boot=20)
    hi = m.predict("L3", {"accelerated_pathway": True, "therapeutic_area": "oncology"})["accelerated"]
    lo_ = m.predict("L3", {"accelerated_pathway": False, "therapeutic_area": "neurology"})["accelerated"]
    assert hi["p"] > 0.5 and lo_["p"] < 0.2
    assert hi["ci90"][0] < hi["p"] + 0.05 and hi["ci90"][1] > hi["p"] - 0.05 and hi["ci90"][0] < hi["ci90"][1]
    # The pathway is a filed-stage input: at L1 it is not used.
    assert m.predict("L1", {"accelerated_pathway": True})["accelerated"]["p"] < 0.5


def test_registry_label_outcomes_only_with_the_fitted_model(monkeypatch):
    from fdaengine.model import registry
    monkeypatch.setenv("FDAENGINE_MODEL", "provisional")
    assert registry.label_outcomes("L3", {}) is None
    monkeypatch.setenv("FDAENGINE_MODEL", "fitted")
    if registry._fitted() is None or registry._labels() is None:
        return                                               # models not built in this checkout
    out = registry.label_outcomes("L3", {"accelerated_pathway": {"value": True}, "anchor_year": {"value": 2018},
                                         "therapeutic_area": {"value": "oncology"}})["outcomes"]
    base = registry.label_outcomes("L3", {"therapeutic_area": {"value": "cardiovascular"},
                                          "anchor_year": {"value": 2018}})["outcomes"]
    assert set(out) == set(lbm.OUTCOMES)
    assert out["accelerated"]["p"] > 0.3 and out["accelerated"]["p"] > 4 * base["accelerated"]["p"]
    assert all(0 <= v["training_rate"] <= 1 and isinstance(v["beats_base_rate"], bool) for v in out.values())
