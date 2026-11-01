from fdaengine.features.build import compute
from fdaengine.sources import pubmed

XML = b"""<?xml version="1.0"?>
<PubmedArticleSet>
 <PubmedArticle><MedlineCitation><PMID>111</PMID><Article>
  <Journal><ISOAbbreviation>Lancet</ISOAbbreviation><JournalIssue><PubDate><Year>2015</Year><Month>Mar</Month></PubDate></JournalIssue></Journal>
  <ArticleTitle>Drug X versus placebo in disease Y: a randomised phase 3 trial</ArticleTitle>
  <Abstract><AbstractText Label="METHODS">Patients were randomised.</AbstractText>
   <AbstractText Label="FINDINGS">The primary endpoint was met in 40% vs 20% of patients (p=0&#183;003).</AbstractText></Abstract>
  <DataBankList><DataBank><DataBankName>ClinicalTrials.gov</DataBankName><AccessionNumberList><AccessionNumber>NCT00000001</AccessionNumber></AccessionNumberList></DataBank></DataBankList>
  <PublicationTypeList><PublicationType>Randomized Controlled Trial</PublicationType></PublicationTypeList>
 </Article></MedlineCitation></PubmedArticle>
 <PubmedArticle><MedlineCitation><PMID>222</PMID><Article>
  <Journal><ISOAbbreviation>J</ISOAbbreviation><JournalIssue><PubDate><Year>2017</Year></PubDate></JournalIssue></Journal>
  <ArticleDate DateType="Electronic"><Year>2016</Year><Month>11</Month><Day>02</Day></ArticleDate>
  <ArticleTitle>Post hoc subgroup analysis of drug X in disease Y</ArticleTitle>
  <Abstract><AbstractText Label="RESULTS">The primary endpoint improved (p=0.04).</AbstractText></Abstract>
  <DataBankList><DataBank><DataBankName>ClinicalTrials.gov</DataBankName><AccessionNumberList><AccessionNumber>NCT00000001</AccessionNumber></AccessionNumberList></DataBank></DataBankList>
  <PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList>
 </Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


def test_parse_dates_links_and_lancet_decimals():
    a, b = pubmed.parse_xml(XML)
    assert a.ncts == ["NCT00000001"] and a.available == "2015-03-31" and a.date_basis == "issue_month"
    assert a.p_value == 0.003 and a.result_basis == "p_value"
    assert b.available == "2016-11-02" and b.date_basis == "epub"
    assert b.p_value is None                                    # secondary analysis: never a trial result


def test_trial_result_comes_from_the_main_paper_only():
    r = pubmed.trial_result(pubmed.parse_xml(XML))
    assert r["pmid"] == "111" and r["p_value"] == 0.003


def test_registry_and_press_release_outrank_the_publication(monkeypatch):
    from fdaengine.features import build
    monkeypatch.setattr(build, "USE_PUBMED", True)
    pub = {"pmid": "1", "available": "2015-03-31", "p_value": 0.2, "z_abs": 1.28, "basis": "p_value", "quote": "q"}
    trial = {"nct_id": "NCT1", "max_phase": 3, "results_status": "none", "primary_completion_date": None,
             "start_date": "2012-01-01", "publication_result": pub,
             "design": {"randomized": True, "single_arm": False, "placebo_controlled": True,
                        "active_controlled": False, "enrollment": 300, "blinding_level": 2}}
    feats, _ = compute("L3", {"sponsor": "X"}, [trial], [], {})
    assert feats["p3_best_pvalue"]["source"] == "pubmed" and feats["p3_primary_endpoint_met"]["value"] == "no"
    pr = [{"event_type": "PHASE3_POSITIVE", "date": "2014-01-01", "value": "0.01", "quote": "met"}]
    feats, _ = compute("L3", {"sponsor": "X"}, [trial], pr, {})
    assert feats["p3_best_pvalue"]["source"] == "timeline" and feats["p3_primary_endpoint_met"]["value"] == "yes"


def test_publications_are_not_features_while_switched_off():
    pub = {"pmid": "1", "available": "2015-03-31", "p_value": 0.2, "z_abs": 1.28, "basis": "p_value", "quote": "q"}
    trial = {"nct_id": "NCT1", "max_phase": 3, "results_status": "none", "primary_completion_date": None,
             "publication_result": pub, "design": {"randomized": True, "single_arm": False, "placebo_controlled": True,
                                                   "active_controlled": False, "enrollment": 300}}
    feats, _ = compute("L3", {"sponsor": "X"}, [trial], [], {})
    assert "p3_best_pvalue" not in feats
