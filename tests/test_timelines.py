from fdaengine.dataset import drugs
from fdaengine.dataset import timelines as tlb

SUZ = "LOG73M21H5"


def _idx():
    return drugs.build_index([
        (SUZ, SUZ, "chemical", "SUZETRIGINE [INN]", "of", False),
        (SUZ, SUZ, "chemical", "VX-548", "cd", False),
        (SUZ, SUZ, "chemical", "JOURNAVX", "bn", False),
    ])


def test_aliases_exclude_brand_names_and_placeholders():
    did = "UNII:" + SUZ
    key_to_drug, names = tlb.build_aliases({did}, {did: ["SUZ", "VX-548 50 mg", "Study drug", "JOURNAVX"]}, _idx())
    assert key_to_drug["SUZETRIGINE"] == did and key_to_drug["VX548"] == did
    assert "JOURNAVX" not in key_to_drug                     # brand names exist only after approval
    assert "STUDYDRUG" not in key_to_drug and "SUZ" not in key_to_drug
    found = tlb.drugs_mentioned("Vertex announced results for VX 548 (suzetrigine) today.", key_to_drug)
    assert found == {did: {"VX548", "SUZETRIGINE"}}


def test_same_company():
    assert tlb.same_company("VERTEX PHARMACEUTICALS INC / MA", "Vertex Pharmaceuticals Incorporated")
    assert tlb.same_company("Merck & Co., Inc.", "Merck Sharp & Dohme LLC")
    assert not tlb.same_company("Pfizer Inc", "Vertex Pharmaceuticals Incorporated")


def test_process_doc_finds_events_for_the_right_programme():
    did = "UNII:" + SUZ
    key_to_drug, names = tlb.build_aliases({did}, {did: ["VX-548"]}, _idx())
    acute = tlb.ProgInfo("PG1", did, "Acute Pain", ["acute", "pain"], "2022-10-03", {"Vertex Pharmaceuticals"})
    dpn = tlb.ProgInfo("PG2", did, "Diabetic Peripheral Neuropathic Pain",
                       ["diabetic", "peripheral", "neuropathic", "pain"], "2024-10-01", {"Vertex Pharmaceuticals"})
    tlb.init_worker(key_to_drug, names, {did: [acute, dpn]})
    text = ("Vertex Announces Positive Results From the VX-548 Phase 3 Program for the Treatment of Moderate-to-"
            "Severe Acute Pain. Both Phase 3 trials of VX-548 in acute pain met their primary endpoints.")
    events, mentions = tlb.process_doc(("0000875320-24-000010_ex99.htm", "2024-01-30", "875320",
                                        "VERTEX PHARMACEUTICALS INC / MA", text))
    assert mentions and {e["programme_id"] for e in events} == {"PG1"}
    assert any(e["event_type"] == "PHASE3_POSITIVE" and e["filer_is_sponsor"] for e in events)


def test_window_and_openfda_shape():
    assert tlb.in_window("2024-01-30", "2022-10-03") and not tlb.in_window("2017-01-01", "2022-10-03")
    app = tlb.openfda_app("219209", "NDA", [{"SubmissionType": "ORIG", "SubmissionNo": "1", "SubmissionStatus": "AP",
                                             "status_date": "2025-01-30", "ReviewPriority": "PRIORITY",
                                             "class_code": "TYPE 1", "class_desc": "Type 1 - New Molecular Entity"}])
    from fdaengine.sources import openfda
    (s,) = openfda.submissions(app)
    assert s.status == "AP" and s.status_date == "2025-01-30" and s.application_number == "NDA219209"


def test_shared_event_goes_to_the_best_matching_indication():
    q = "The FDA accepted the BLA for fremanezumab for the preventive treatment of migraine in adults."
    ev = [{"programme_id": p, "drug_id": "D", "source_id": "doc", "event_type": "NDA_ACCEPTED",
           "event_date": "2017-12-18", "quote": q} for p in ("MIG", "CLUSTER")]
    kw = {"MIG": ["migraine"], "CLUSTER": ["cluster", "headache"]}
    assert [e["programme_id"] for e in tlb.best_indication(ev, kw, {"NDA_ACCEPTED"})] == ["MIG"]
    # fragmented indications ('Osteoarthritis' / 'Osteoarthritis, Hip') and sentences that match no indication keep
    # every programme
    kw_frag = {"MIG": ["migraine"], "CLUSTER": ["migraine", "adults", "chronic"]}
    assert len(tlb.best_indication(ev, kw_frag, {"NDA_ACCEPTED"})) == 2
    assert len(tlb.best_indication(ev, {"MIG": ["asthma"], "CLUSTER": ["gout"]}, {"NDA_ACCEPTED"})) == 2
    # other event types are untouched
    assert len(tlb.best_indication(ev, kw, {"PDUFA_DATE"})) == 2
