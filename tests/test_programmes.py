from fdaengine.dataset import drugs
from fdaengine.dataset import programmes as prog
from fdaengine.features.indication import therapeutic_area

SUZ, HYD, APAP = "LOG73M21H5", "HYD000000X", "APAP00000X"


def _idx():
    rows = [
        (SUZ, SUZ, "chemical", "SUZETRIGINE [INN]", "of", False),
        (SUZ, SUZ, "chemical", "VX-548", "cd", False),
        (SUZ, SUZ, "chemical", "JOURNAVX", "bn", False),
        ("SALT00000X", HYD, "chemical", "HYDROCODONE BITARTRATE", "cn", False),   # salt → active moiety
        (HYD, HYD, "chemical", "HYDROCODONE", "cn", True),
        (APAP, APAP, "chemical", "ACETAMINOPHEN", "cn", True),
        ("ULI000000X", "ULI000000X", "chemical", "ULIPRISTAL", "cn", True),
        ("ULIACET00X", "ULI000000X", "chemical", "ULIPRISTAL ACETATE", "cn", False),
        ("ULIACET00X", "ULI000000X", "chemical", "ESMYA", "bn", False),
        ("ACETATE00X", "ACETATE00X", "chemical", "ACETATE", "cn", True),
    ]
    return drugs.build_index(rows)


def test_resolver_handles_codes_doses_salts_and_combinations():
    idx = _idx()
    assert drugs.resolve_string("VX 548", idx)[0] == {SUZ}
    assert drugs.resolve_string("Suzetrigine 50 mg tablets", idx)[0] == {SUZ}
    assert drugs.resolve_string("Hydrocodone bitartrate/acetaminophen", idx)[0] == {HYD, APAP}
    assert drugs.resolve_string("Placebo (matched to SUZ)", idx)[0] == frozenset()
    assert drugs.resolve_string("SUZ", idx)[0] == frozenset()            # abbreviation: resolved via alias union
    assert drugs.resolve_string("Ulipristal acetate Esmya", idx)[0] == {"ULI000000X"}   # not the acetate ion
    assert drugs.resolve_string("Acetate buffer", idx)[0] == frozenset()
    assert idx.display[SUZ] == "SUZETRIGINE"                             # INN preferred over the brand name


def test_alias_union_never_merges_two_known_drugs():
    u = drugs.AliasUnion()
    assert u.union("UNII:" + SUZ, "NAME:SUZX")
    assert u.find("NAME:SUZX") == "UNII:" + SUZ
    assert not u.union("NAME:SUZX", "UNII:" + HYD)                       # would join suzetrigine and hydrocodone
    assert u.find("UNII:" + HYD) == "UNII:" + HYD


def _t(nct, start, leaves, texts, anc=()):
    return prog.TrialCond(nct, start, leaves, [prog.norm_text(t) for t in texts], set(anc))


def test_suzetrigine_grouping():
    """Acute-pain pivotal trials + a single-arm safety study coded only as 'Pain' form one programme; the DPN
    trials (no MeSH term) form another."""
    pain, acute = "D010146", "D059787"
    trials = [
        _t("NCT05553366", "2022-10-03", [acute], ["Acute Pain"], [pain]),
        _t("NCT05558410", "2022-10-10", [acute], ["Acute Pain"], [pain]),
        _t("NCT05661734", "2023-01-09", [pain], ["Pain"]),
        _t("NCT06628908", "2024-10-01", [], ["Diabetic Peripheral Neuropathic Pain"]),
        _t("NCT06696443", "2024-12-18", [], ["Diabetic Peripheral Neuropathic Pain"]),
    ]
    groups = prog.group_trials(trials, generic={pain})
    as_sets = sorted(sorted(g.trials) for g in groups)
    assert as_sets == [["NCT05553366", "NCT05558410", "NCT05661734"], ["NCT06628908", "NCT06696443"]]
    acute_group = next(g for g in groups if "NCT05661734" in g.trials)
    assert acute_group.attached["NCT05661734"] == "generic_attached"
    assert acute_group.specificity == "specific"


def test_secondary_condition_does_not_merge_programmes():
    pso, psa = "D011565", "D015535"
    trials = [_t("A", "2015-01-01", [pso], ["Psoriasis"]),
              _t("B", "2015-06-01", [psa, pso], ["Psoriatic Arthritis", "Psoriasis"])]
    assert len(prog.group_trials(trials, generic=set())) == 2


def test_condition_text_matching():
    n = prog.norm_text
    assert prog.texts_match(n("Moderate to Severe Glabellar Lines"), n("Glabellar Frown Lines"))
    assert prog.texts_match(n("HER2-positive Metastatic Breast Cancer"), n("Breast Cancer, HER2-positive"))
    assert not prog.texts_match(n("Pain"), n("Diabetic Peripheral Neuropathic Pain"))
    assert not prog.texts_match(n("Acute Pain"), n("Chronic Pain"))


def test_salt_without_gsrs_moiety_link_maps_to_parent():
    idx = drugs.build_index([("GEM0000000", "GEM0000000", "chemical", "GEMCITABINE", "cn", True),
                             ("GEMHCL0000", "GEMHCL0000", "chemical", "GEMCITABINE HYDROCHLORIDE", "cn", True),
                             ("NACL000000", "NACL000000", "chemical", "SODIUM CHLORIDE", "cn", True)])
    assert idx.lookup("Gemcitabine HCl") | idx.lookup("gemcitabine hydrochloride") == {"GEM0000000"}
    assert drugs.resolve_string("0.9% Sodium Chloride", idx)[0] == frozenset()


def test_episodes_split_after_seven_years():
    eps = prog.split_episodes({"a": "2005-01-01", "b": "2008-01-01", "c": "2014-01-01", "d": None})
    assert eps == [["a", "b", "d"], ["c"]]


def test_indication_link_levels():
    pain, acute = "D010146", "D059787"
    p2 = _t("P2", "2021-07-19", [acute], ["Acute Pain"], [pain])
    assert prog.indication_link(p2, {acute}, {pain}, {"acute pain"}, {pain}) == "exact"
    p2_broad = _t("P2b", "2021-07-19", [pain], ["Pain"])
    assert prog.indication_link(p2_broad, {acute}, {pain}, {"acute pain"}, {pain}) == "related"
    other = _t("P2c", "2022-12-20", [], ["Diabetic Peripheral Neuropathy"])
    assert prog.indication_link(other, {acute}, {pain}, {"acute pain"}, {pain}) is None


def test_therapeutic_area_precedence():
    assert therapeutic_area(["Carcinoma, Non-Small-Cell Lung"], ["Neoplasms", "Lung Diseases"], []) == "oncology"
    assert therapeutic_area(["Acute Pain"], ["Pain", "Neurologic Manifestations"], []) == "pain"
    assert therapeutic_area([], [], ["Diabetic Peripheral Neuropathic Pain"]) == "pain"
    assert therapeutic_area([], [], ["something unheard of"]) == "other"
