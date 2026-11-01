# Gold-set findings: systematic pipeline errors (v2 review, 2026-10-30)

New systematic errors found while verifying the gold set against primary sources (FDA letters, Drugs@FDA, Purple
Book, sponsor 8-K/6-K filings, ClinicalTrials.gov). Each has an example programme. Not fixed here (evaluation only).
Errors already recorded in verdict notes by the first pass (negated label text, generic-keyword label matches,
line-of-therapy, rule order, comparator-arm groupings, protein-name resolution) are not repeated unless the review
found a new consequence.

## Label attribution (Y_PROG)

1. **Withdrawn indications are invisible to current-label attribution.** The approval is attributed only if
   today's openFDA label still lists the disease, so accelerated approvals later withdrawn are lost.
   Example: atezolizumab / urothelial carcinoma (PGc729490263): Tecentriq's 2016-05-18 urothelial approval is
   no longer on the label, so the pipeline codes failed.
2. **Discontinued brands without a current openFDA label lose their supplements.** Example: meloxicam / RA
   (PG6697866043): Mobic S-004 (RA, 2004-07-16) is in Drugs@FDA but was not credited (no_approval_found).
   Ropinirole / Requip XL (PGed8239fb11) is the same pattern for an original NDA.
3. **The Drugs@FDA bulk file drops some withdrawn applications entirely.** Example: lorcaserin (PG08f029b545):
   Belviq NDA 022529 (approved 2012-06-27) is absent from Applications/Products/Submissions; only NDA 209464 lists
   lorcaserin. Approvals of withdrawn products need a second source (FDA letters, sponsor filings).
4. **Very recent supplement approvals are missed while the openFDA label lags.** Example: dostarlimab / rectal
   cancer (PG81a6ee76e7): Jemperli S-015 (dMMR rectal, 2026-10-19) is in the 2026-10-28 Drugs@FDA file but the
   programme stays censored.
5. **The wrong supplement's date is used when several supplements share a label that covers the disease.** The
   pipeline picks a supplement whose current label mentions the disease, not the one whose approval letter adds the
   use. Examples: lacosamide / POS (PG9d492e193b) dated 2020-11-16 instead of S-049 2021-10-14; tenofovir
   alafenamide / HBV (PG3a32d35c10) dated 2018-07-03 (labeling update) instead of the 2016-11-10 original;
   ipilimumab / RCC (PG610f792e8b) dated 2020-03-10 (the HCC supplement) instead of S-094 2018-04-16.
6. **Code-name collisions now also hide real approvals.** Example: oveporexton (TAK-861) / narcolepsy
   (PG8f648463d0) resolved to the CDK9 protein, so Orzeyful's approval (2026-08-05) was never matched and the
   programme stays censored. The same TAK- collision affects PG370996f514.

## Next action (Y_NEXT)

7. **CRLs and approvable letters before the FDA CRL database are mostly missed.** They exist only in sponsor
   press releases, which the parser does not turn into CRL dates. Examples: cethromycin (PGd3a1b0733c) CRL
   2009-07-31 in an 8-K; efaproxiral (PG133a0bfef1) approvable letter 2004-06-02; Preos (PG23392f0ac4) approvable
   letter 2006-03-09; lorcaserin (PG08f029b545) CRL 2010-10-22; Oleptro (PGae1d17b88b) CRL 2009-07-17;
   tigecycline CAP (PGd18d7e6f95) action letter 2008-05-27. All are coded no_action_found.
8. **CRLs are attached at drug level, across indications.** Example: canakinumab / SJIA (PG64f178ec28) gets the
   2011 gout-sBLA CRL as its next action (truth: approved 2013-05-09). Sotagliflozin / T2D (PGb61a98e6c2) gets the
   type 1 diabetes CRL.
9. **Y_NEXT inherits every Y_PROG attribution miss.** When the approval is not attributed, the next action
   becomes no_action_found. Examples: deoxycholic acid / Kybella (PGe778a2603e), lurbinectedin (PG4f441088e8),
   erlotinib / pancreatic (PG1c3a9414d1), Palforzia (PG347d24387c).
10. **FDA actions before the PDUFA date put L5 after the decision.** L5 = PDUFA − 7 days is not checked against
    an earlier action. Example: nivolumab / cHL (PGeb3d5f7cce): the 1L cHL sBLA was approved 2026-03-20, but L5 is
    2026-04-01 and the pipeline still says pending. The landmark validity check catches this only when the
    approval is attributed to the programme.

## Level dates (L2 / L3 / L5)

11. **L2 lands on a later readout when the first Phase 3 readout was a miss, or came from an earlier trial.**
    16 of 46 reviewed L2 dates are late. Examples: canakinumab / NSCLC (PG6fc3ac5aee) uses the CANOPY-1 miss
    (2021-10) and skips the CANOPY-2 miss (2021-03-09); vilazodone (PG581ce78bf5), lubiprostone / OIC (PGfce9ce6413),
    tanezumab (PG62778fd9da) and trazodone OAD (PGae1d17b88b) take the second or third positive trial.
12. **L3 often comes from a quarterly recap rather than the first acceptance release.** Examples: ertugliflozin
    (PG476b8e5ea5) is dated from Q1 2017 earnings (acceptance announced March 2017); G/P (PG4027f2a2a2) is 2017-04-28
    vs 2017-02-08; Kybella (PGe778a2603e) is 2014-08-08 vs 2014-07-11.
13. **PDUFA extensions are not parsed, so L5 uses the superseded goal date.** Examples: fampridine (PG7fb8c86cff)
    2009-10-22 → extended to 2010-01-22; rilonacept (PG13f824009c) 2007-11-29 → 2008-02-29. The cut-off is then
    too early (no leakage, but a wrong landmark).

## L3 follow-up (2026-11-01)

Why only ~63% of novel-drug L3 rows end in approval within 3 years (literature NDA→approval ~85–90%). A review of
30 L3 rows that failed or were approved > 3 years after the cut-off found about half genuine (CRL or withdrawal:
reproxalap, abicipar, omecamtiv, drisapersen, tanezumab), about a quarter missed approvals, and the rest filings
for another indication or country. Fixed in this pass (03 rerun; gold set unchanged, one wrong L3 removed):
non-US filings that named an application acronym ('The NDA in China … accepted by the NMPA'); capitalised
'Investigational New Drug application' read as an NDA acceptance; 'standard review' matched in comparisons
('accelerates the standard review time'); a filing parsed for several programmes of a drug now stays only with those
whose indication its own sentence names (209 events moved). Still open:

14. *(Mostly fixed 2026-11-01 by the floor change in item 15; relatlimab and deoxycholic acid remain.)*
    **Approvals of multi-programme drugs without a label match go to no programme.** Example: adagrasib / NSCLC
    (L3 2022-03-01): Krazati (2022-12-12) is "no_approval_found" because the drug also has a rectal programme.
    Relatlimab / melanoma (Opdualag 2022-03-18) and deoxycholic acid / submental fat (Kybella 2015-04-29) likewise.
15. *(Fixed 2026-11-01: the floor now applies only when another programme of the drug had already finished its
    Phase 3; gold set novel Y_PROG 90.1% → 91.4%.)* **The "approval follows the programme's own Phase 3" floor uses registry completion dates, which are often
    still in the future for trials approved on interim data.** Example: exagamglogene autotemcel / SCD and TDT
    (Casgevy, BLA 125787, 2023-12-08, matched in the Purple Book) is coded failed.
16. **Some CBER licences have proper names that share no words with the INN.** Example: efanesoctocog alfa
    (Altuviiio, "antihemophilic factor (recombinant), Fc-VWF-XTEN fusion protein-ehtl") is not matched.
17. **Cross-indication filings that share a disease word still attach.** Examples: dostarlimab endometrial filing on
    the ovarian programme ("cancer"), lecanemab early-AD filing on the preclinical-AD programme.

## Late-level supplement (2026-11-01, 60 filed novel programmes, `why` = "late-level supplement")

Drawn by pipeline Y_LEVEL (36 not approved within 3 years of L3, 24 approved) and reviewed against Drugs@FDA, the
Purple Book, the CRL database and sponsor filings. Scored in `gold_report.json` → `y_level` (with the main sample's
L3/L5 rows). Result before the fixes below: Y_LEVEL at L3 83% right (PPV 88%, NPV 78%; errors roughly cancel, so the
corrected L3 approval rate equals the pipeline's), but the filing event behind L3 was right only 67% of the time.

Fixed in this pass:

18. **Filing milestones from multi-drug documents.** A PDUFA date or acceptance in a pipeline round-up was attached
    to whichever programme drug was named in the preceding sentences: tislelizumab got Brukinsa's WM filing,
    ozanimod a fedratinib goal date, trofinetide pimavanserin's sNDA, ublituximab umbralisib's FL filing. Script 03
    now requires the drug in the event sentence for filing events from documents that mention more than one
    programme drug (L3 evidence 67% → 76%, L5 65% → 75%; L3 programmes 756 → 584, L5 295 → 187).
19. **"The FDA accepted the notion of …"** (enclomifene) read as an NDA acceptance. Parser guard added.
20. **`strip_dose` deleted every digit** (the root cause behind item 6 and the code-prefix flag; after the fix code-prefix flags fell 59 → 2, drug resolution 88% → 91.7%, novel Y_PROG 91.4% → 93.0%): 'HTX-011' → 'HTX'
    → histrionicotoxin (Zynrelef never matched), 'AR101' → panobacumab, 'CSL830' → conestat alfa. Only standalone
    numbers are stripped now; scripts 02–05 rerun. Programmes whose drug changed got new ids; their gold rows are
    kept with `detached` = yes and left out of scoring.

Still open (one or two cases each):

21. **Supplements for a sibling indication sharing a disease word** are credited to the wrong programme: mitapivat's
    thalassemia approval (Aqvesme) to the sickle cell programme ("anemia"); Orlynvah (uUTI) to sulopenem's
    intra-abdominal programme ("infections").
22. **Label wording that shares no words with the indication label** blocks attribution: Giapreza ("septic or other
    distributive shock") for "catecholamine-resistant hypotension".
23. **Withdrawn indications** (item 1) also affect late levels: Imfinzi's 2017 urothelial approval.
24. **A metabolite resolved instead of the drug** makes an old drug look novel: beclomethasone dipropionate →
    beclomethasone 17-monopropionate (no approval history). Audit: 10 of 4,425 novel programmes name an ingredient
    approved before their anchor.
