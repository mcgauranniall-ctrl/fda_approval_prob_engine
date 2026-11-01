from fdaengine.dataset import drugs

CDK9, TAF, CARB, CTP = "CDK9000000", "TAF0000000", "CARB000000", "CTP0000000"
KEYS = {CDK9: {"CYCLINDEPENDENTKINASE9", "TAK"}, TAF: {"TENOFOVIRALAFENAMIDE", "TAF", "GS7340"},
        CARB: {"CARBAPENEM"}, CTP: {"CTP", "CYTIDINETRIPHOSPHATE"}}
DISPLAY = {CDK9: "CYCLIN-DEPENDENT KINASE 9", TAF: "TENOFOVIR ALAFENAMIDE", CARB: "CARBAPENEM", CTP: "CTP"}


def flags(drug_id, rows):
    return drugs.resolution_flags(drug_id, rows, KEYS, DISPLAY)


def test_target_protein_reached_through_a_code_name_prefix():
    assert flags(f"UNII:{CDK9}", [("TAK-861", "normalised"), ("TAK-861", "normalised")]) == ["not_a_drug", "code_prefix"]
    assert flags(f"UNII:{CTP}", [("CT-P53", "normalised")])[:2] == ["not_a_drug", "code_prefix"]


def test_real_drug_resolved_exactly_is_clean():
    assert flags(f"UNII:{TAF}", [("TAF|TDF", "exact"), ("GS-7340", "normalised")]) == []


def test_vehicle_weak_match_and_descriptive_names():
    assert "vehicle" in flags(f"UNII:{TAF}", [("Castor Oil (Placebo)", "other_name_exact")])
    assert flags(f"UNII:{CARB}", [("Vabomere", "other_name_word")]) == ["not_a_drug", "weak_match"]
    assert flags("NAME:STANDARDADJUVANTENDOCRINETHERAPY", [("Standard Adjuvant Endocrine Therapy", "name_only")]) \
        == ["descriptive_name"]
    assert flags("NAME:HDVINSULIN", [("HDV-insulin", "name_only")]) == []


def test_placebo_arm_name_and_abbreviations_do_not_flag_real_drugs():
    assert flags(f"UNII:{TAF}", [("TAF|TAF Placebo", "exact")]) == []
    assert flags(f"UNII:{TAF}", [("BD MDI 320 ug", "other_name_normalised")]) == []      # abbreviation only
    assert flags(f"UNII:{TAF}", [("Stribild", "alias")]) == []                           # brand alias


def test_strip_dose_keeps_code_names():
    # Digits joined to letters or a hyphen are part of a code name; only standalone numbers and doses go.
    assert drugs.strip_dose("TAK-861") == "TAK-861"
    assert drugs.strip_dose("AR101") == "AR101"
    assert drugs.strip_dose("MK-0431 100 mg") == "MK-0431"
    assert drugs.strip_dose("Ulipristal acetate 30") == "Ulipristal acetate"
