from fdaengine.dataset import labels as lab
from fdaengine.dataset.purplebook import Bla, core_tokens, match


def bla(no, applicant, proper, brand, date):
    return Bla(no, applicant, proper, brand, date, core_tokens(proper))


BLAS = [
    bla("125614", "GlaxoSmithKline Biologicals", "Zoster Vaccine Recombinant, Adjuvanted", "SHINGRIX", "2017-10-20"),
    bla("103475", "GlaxoSmithKline Biologicals", "Hepatitis A Vaccine", "HAVRIX", "1995-02-22"),
    bla("103239", "GlaxoSmithKline Biologicals", "Hepatitis B Vaccine (Recombinant)", "ENGERIX-B", "1989-08-28"),
    bla("125127", "GlaxoSmithKline Biologicals", "Influenza Vaccine", "FLUARIX", "2005-08-31"),
    bla("125248", "GlaxoSmithKline Biologicals", "Influenza Virus Vaccine, H5N1", "", "2013-11-22"),
    bla("125694", "Novartis Gene Therapies", "onasemnogene abeparvovec-xioi", "ZOLGENSMA", "2019-05-24"),
]


def gsk(applicant):
    return "glaxo" in applicant.lower()


def test_core_tokens_keep_the_distinguishing_words():
    assert core_tokens("Zoster Vaccine Recombinant, Adjuvanted") == {"zoster"}
    assert core_tokens("Hepatitis A Vaccine") != core_tokens("Hepatitis B Vaccine (Recombinant)")
    assert core_tokens("onasemnogene abeparvovec-xioi") == {"onasemnogene", "abeparvovec"}


def test_vaccine_matches_by_antigen_words_and_holder():
    got = match("UNII:X", ["RECOMBINANT VARICELLA ZOSTER VIRUS GLYCOPROTEIN E ANTIGEN"], gsk, BLAS, {})
    assert [b.bla for b in got] == ["125614"]
    hep_a = match("UNII:Y", ["HEPATITIS A VIRUS STRAIN HM175 ANTIGEN (FORMALDEHYDE INACTIVATED)"], gsk, BLAS, {})
    assert [b.bla for b in hep_a] == ["103475"]
    # Another company's trial of the same antigen is not GSK's licence.
    assert match("UNII:X", ["varicella zoster glycoprotein E"], lambda a: False, BLAS, {}) == []


def test_most_specific_licence_wins():
    got = match("UNII:Z", ["Influenza virus vaccine H5N1 (split virion, inactivated, adjuvanted)"], gsk, BLAS, {})
    assert [b.bla for b in got] == ["125248"]


def test_resolved_drug_id_matches_regardless_of_holder():
    got = match("UNII:MLU3LU3EVV", ["AVXS-101"], lambda a: False, BLAS, {"125694": "UNII:MLU3LU3EVV"})
    assert [b.bla for b in got] == ["125694"]


def test_cber_licence_goes_to_each_programme_under_way():
    act = lab.Action("2017-10-20", "125614", "ORIG", "CBER", True, None)
    f = lab.Facts("PG1", "2011-08-01", False, n_rival_programmes=2, actions=[act], cber=True)
    out = lab.programme_label(f, "2026-10-29")
    assert (out["y_prog"], out["label_source"], out["label_confidence"]) == (1, "purplebook_cber", "low")
