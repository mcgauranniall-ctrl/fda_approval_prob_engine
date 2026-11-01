# FDA Approval Probability Engine v2 — Build Plan

> **Purpose:** a general-purpose, statistically validated forecast of whether (and **when**) a drug's Phase 3
> programme will end in FDA approval, sold to more than one kind of customer:
> - **Payers / health insurers** (e.g. large US insurers such as UnitedHealthcare): forecast which pipeline drugs will
>   reach the market, when, with what label, and at roughly what cost, so premiums, budgets and formulary plans can be
>   set before approval is guaranteed.
> - **Investors**: know before the market prices it in. The stock-market strategy test is the **last thing built**.
>
> **How it works:** you type a drug + indication into a Python UI and **choose the information level** to forecast
> at (Phase 2 only, after the Phase 3 readout, after filing, after the AdCom, just before the PDUFA date, or latest
> available). The engine then:
> 1. downloads every public document released up to that point into a new case folder;
> 2. runs a **headless Claude Code prompt** (the APEX pattern) that reads them and writes structured, cited data;
> 3. puts that data through a statistical model **trained on historical programmes at the same level**;
> 4. produces a v1-style report: the probability with a real interval, a confidence rating, risk flags, drivers,
>    comparable cases and the probability path across levels, plus an **audience panel**: payer (approval-by-date,
>    expected label, budget impact) or investor (next catalyst, typical stock reaction, model vs market).

---

## Decisions log

| # | Decision | Resolution |
|---|---|---|
| D1 | What is being predicted | **P(the Phase 3 programme for this drug × indication ends in FDA approval).** Also: **P(approved by a given date)** (timing), **expected label characteristics** (boxed warning, REMS, accelerated vs full approval, label narrower than the trial population), and from the filing level on, **P(approval at the next FDA action / PDUFA date)** |
| D2 | Allowed inputs | **Anything public before the FDA decision**, filtered to the selected level's cut-off date. Excluded: the decision itself and documents the FDA releases after it (approval letters, review packages, final labels) |
| D3 | Forecast level | **User-selectable** (§2): Phase 2 complete → Phase 3 readout → Filed → Post-AdCom → Pre-PDUFA → Resubmission after CRL → "Latest available" |
| D4 | Application types | All NDA/BLA pathways, **focused on Rx prescription drugs**; Rx-subgroup performance reported separately |
| D5 | How documents are read | **No Anthropic API.** Python downloads deterministically, then launches `claude -p` with prompt files (the APEX `run-review.ps1` pattern), **only for the case you're analysing, started from the UI** |
| D6 | Training data | Built from **structured sources + rule-based parsers only**: no scheduled Claude batches. An optional script lets you run Claude extraction on a sample yourself, if you choose to |
| D7 | UI | **Streamlit**; the report is a self-contained v1-style HTML page embedded in the app |
| D8 | Applications & build order | The **core engine is audience-neutral** and is validated on statistical grounds alone (calibration, discrimination, timing accuracy). Applications sit on top as modules: **payer module first, investor module last**. The stock-market strategy test is the final phase of the whole build |
| D9 | Priority (user decision 2026-11-01) | **The payer use case (UnitedHealthcare-style pipeline and budget forecasting) is the main priority.** No work on stock-market capabilities (investor module, event study, market backtest) until the user asks for it. The roadmap (§14) puts what payers need first: approval timing, new-indication (supplement) labels, label breadth, then the payer module |

---

## Build status (updated 2026-11-01)

**Done: the "fast first win" slice (P0, P1, P2, P7, P8), running on the provisional model.** It was tested end to
end on suzetrigine (acute pain) in live and backtest modes at L1–L5, with 16 offline unit tests.

Deviations from the plan, decided during the build:
- **Claude runs are read-only.** `--tools Read,Glob,Grep` + `--json-schema`; Python writes every file. This is
  stricter than the planned scoped-write permissions.
- **One Claude run per case, not two.** Extraction and analyst flags share a single run; `analyst_review.md` is
  merged into `extract_case.md`. The memo is a deterministic template for now (`write_memo.md` not built yet).
- **ClinicalTrials.gov record history** returns 403 from the public API. Registration fields therefore use the
  current version (flagged in the manifest); results and documents posted after the cut-off are stripped.
- **Brand names and future dates are scrubbed from the case folder.** A brand name like JOURNAVX only exists after
  approval, so it would leak the outcome. Level dates and backtest outcomes live in `<case>.meta.json`, outside
  Claude's working directory.

Known gaps:
- **Standalone press releases from large-cap sponsors.** These are often not filed with the SEC (e.g. Vertex's Phase
  3 topline), so some Phase 3 p-values are missing at L2. Fix: a sponsor investor-relations fetcher, or the gap-fill
  run (§9.1).
- PubMed, AdCom pages, the CRL database, the orphan database and Open Targets clients are not built yet.
- OCR for scanned PDFs is not wired in (scanned pages are flagged in the manifest).

**P3 historical dataset (updated 2026-10-30): scripts 01–05 built and run, gold set scored.**

| Script | Output |
|---|---|
| 01 | 98,020 industry Phase 1–3 drug/biologic trials (ClinicalTrials.gov API), Drugs@FDA, openFDA (labels, drugsfda, CRL, UNII, GSRS), Purple Book; sha256 manifest |
| 02 | 13,102 drug × indication programmes (7-year episodes) from 22,931 seed industry Phase 3s (was 12,317 before the 2026-11-01 `strip_dose` fix); 98% of Drugs@FDA products resolved to GSRS moieties |
| 03 | EDGAR full-text crawl by event phrase × year (8-K/6-K, 2001–): 104,373 press-release documents; rule-based events; levels and cut-offs per programme |
| 04 | Y_PROG (competing-risks coding, IPCW-ready), Y_NEXT, Y_P3, `label_source`, `label_confidence`, `label_tier`; cohort 6,154 programmes, primary population (novel drugs) 4,713 (922 approved, 2,148 failed, 1,401 open, 242 excluded), 30.0% approved among resolved; 87 `resolution_suspect` programmes dropped (160 before the `strip_dose` fix) |
| 05 | 10,695 landmark rows (7,256 in cohort after the 2026-11-01 fixes); every row passes the no-post-cut-off-input check |
| Gold set | 212 programmes (`data/gold/`): the main stratified sample (152; 2 detached by the 2026-11-01 rebuild, 12 re-attached by seed trials) plus a **late-level supplement** of 60 filed novel programmes (2026-11-01). Every outcome checked against primary sources (`evidence` + `verified`; Claude-reviewed, not yet spot-checked by a person). After the 2026-11-01 fixes: Y_PROG 93.0% right on novel drugs in the training population (n=71; approved share 49.3% pipeline vs 50.7% truth), 82.7% overall; drug resolution 91.7%; Y_LEVEL at L3 85% right (n=80), L5 91% (n=33); filing event behind L3 right 73%; post-decision landmarks 3% (L3) / 6% (L5). Systematic errors and fixes: `data/gold/findings.md` |

Deviations, decided during the P3 build:
- **ClinicalTrials.gov API v2 instead of AACT** (same JSON as live cases, so `structure_trial()` is shared).
- **Default population = US-relevant programmes** (a Phase 3 with a US site or flagged FDA-regulated; user decision
  2026-10-30). Non-US novel programmes are 5% approved and mostly never seek FDA approval. Site lists are the
  registry's current ones, so this is a cohort filter, never a feature.
- **Cohorts:** `novel` and `lifecycle_holder` (sponsor holds the NDA/BLA, so a supplement can approve the new use)
  are modelled; `lifecycle_other` (generics, 505(b)(2), biosimilars by another holder) and `lifecycle_cber`
  (licensed CBER products: the Purple Book has no indication supplements) are flagged out.
- **CBER products (vaccines, blood, cell/gene therapy) are labelled from the Purple Book** (user decision), since
  Drugs@FDA covers CDER only. A licence goes to every programme of the product under way before it unless a label
  exists that does not cover the indication. Unmatched CBER products with unresolved names are excluded.
- **Approval attribution uses the current FDA label's Indications section** (openFDA), matched against the MeSH
  indication and each trial's own condition text. Labels are today's versions; ~20% of marketed NDA/BLAs have no
  harmonised openFDA label.
- **Primary population = novel drugs** (`in_primary`). Lifecycle labels rest on today's label mentioning the disease, which the gold set found right 63% of the time (line of therapy, other manifestations): a secondary analysis; supplement-only approvals are `label_tier = sensitivity`.
- **Mis-resolved drugs are dropped** (`resolution_suspect`, user decision 2026-10-30). A resolved "drug" that is a
  target protein, class term or bare acronym (often reached by stripping a code name's digits: 'TAK-861' → 'TAK' →
  CDK9), a vehicle, or a regimen description can never match its approval, so its label is forced to failed. Rules
  use names and trial text only (no approval status; brand names only if approved before the anchor), so the drop
  is outcome-neutral. Gold set: 10 of 19 wrong drugs caught, no false alarms; novel-drug Y_PROG 87.0% → 90.1%.
  `weak_match` (substance name absent from fuzzy-resolved trial text) is recorded but not dropped (~50% precise).
  Root cause repaired 2026-11-01 (`strip_dose` now keeps code names); the filter now drops 87 programmes.
- **Press-release-only approvals are not labels** (`edgar_only_unverified`, excluded from Y_PROG): ~40% precise on
  review. Press approvals count only when they coincide with an FDA action.
- **Event dates are conservative:** "In March" / "in Q1" give the end of the period (capped at the filing date),
  so a cut-off never precedes the announcement. Phase 3 readouts and regulatory events dated before a programme's
  first Phase 3 are dropped (they belong to an earlier episode); every CRL review cycle is kept.
- **Parser precision rules:** the drug must be named within 250 characters of the event phrase (or in the
  preceding sentences of a short sentence); non-US agency filings are ignored; approvals must be factual (not
  conditional or negated).
- **Leakage handling without registration history:** trials count only once first posted; results only once
  posted; the *actual* enrolment of a trial still running at the cut-off is withheld; the FDA CRL database
  (published 2025) is a label source only.

Known limitations of the training data:
- Drug-name resolution is outcome-dependent (GSRS knows approved drugs' codes better), so `drug_resolution` and
  `cutoff_estimated` (whether a topline release was ever found) must never be features.
- Y_NEXT is unlabelled for ~half of L3/L5 rows (no FDA action found within 2 years); L4 has 34 cohort rows.
- Indication fragmentation and moiety mismatches remain (e.g. ZS-9 resolved as "ZIRCONIUM"); the gold set
  measures their rate. Drugs@FDA `SponsorName` is the current holder, so sponsor prior approvals carry some
  acquisition look-ahead.
- Not yet downloaded: Open Targets (H5 genetic support), orphan designations database, AdCom pages, inspections.

**P4 statistical inference (2026-11-01): scripts 06–07 built and run** (`reports/eda.html`, `reports/inference.html`;
CSVs in `data/processed/`: univariate, missingness, hypotheses, stability, cuminc, benchmark). Target **Y_LEVEL**
(below). Population: novel drugs with final outcomes (7-year window closed at L1–L2; 3-year horizon passed or approved
from L3 on), so no censoring correction is needed; L2 restricted to rows with a press-release-dated readout
(n=345), the live-case situation. Figures are from the final 2026-11-01 rebuild.

| Script | Output |
|---|---|
| 06 | Univariate screen per level (Fisher / χ² / Mann–Whitney + logistic OR, BH FDR q=0.05): 25 of 229 tests pass in the primary screen; no direction flips in the all-labelled sensitivity. Missingness diagnostic: press-release fields are present far more often for approved programmes (L2 Phase 3 met: 43% approved when present vs 9% when missing), so their missingness is never a feature |
| 07 | Hypothesis table (GLM adjusted for sponsor size, sponsor prior approvals, TA prior rate, era, pivotal trials, partners; SEs clustered by drug; Firth when a cell < 5); stability selection (L1 logistic, 500 half-subsamples, keep ≥ 0.7); Aalen–Johansen P(approved within t) per level; benchmark of the provisional model on the same rows |

Findings:
- **Supported:** H1 Phase 2 met (adj. OR 6.4) and Phase 2 z; Phase 3 met at L2 (OR 14.7); H7 ≥2 pivotal trials
  (OR 2.3); H11 an AdCom being scheduled is negative at L3 (OR 0.35); H15 large vs small/private sponsor (OR 3.0 at L1,
  7.9 at L2); H16 later era raises odds at L2.
- **Contradicted (to review, not built into the model):** randomised Phase 2 (OR 0.62) and single-arm Phase 2
  (OR 1.44) at L1; having a registered Phase 2 at L1 (OR 0.70, p = 0.049); Phase 3 results already posted at L2
  (OR 0.23). Verdicts near p = 0.05 moved with label corrections during 2026-11-01 (Fast Track at L3, H10): fragile.
- **Insufficient data:** H4 assurance, H6 Phase 3 z, H8 safety imbalances, H11 AdCom vote (coverage 2–17%: mostly
  Claude-only fields in live cases). Not testable yet: H2, H5 genetics, H13, H14, H18.
- **The provisional model over-predicts and has little skill.** Mean predicted vs observed Y_LEVEL: L1 50% vs 37%,
  L2 72% vs 50%, L3 90% vs 68%, L5 90% vs 65%; AUROC 0.58–0.67; Brier no better than the base rate.

**Data readiness (2026-11-01): `scripts/08_data_readiness.py`** (`reports/data_readiness.html`,
`data/processed/label_error.json`). Label error from the gold set, re-weighted to the training population, and the
statistical capacity per level:

| Level | Rows | Approved | Sensitivity / specificity | Corrected approval rate (95%) | Parameters (10 EPV) | Min. detectable OR (binary 30%) | Verdict |
|---|---|---|---|---|---|---|---|
| L1 | 1,254 | 462 | 0.87 / 0.97 | 40% (34–48) | 46 | 1.4 | ready |
| L2 | 345 | 172 | 0.92 / 0.94 | 51% (45–58) | 17 | 1.9 | ready |
| L3 | 281 | 191 | 0.91 / 0.78 | 67% (57–73) | 9 | 2.2 | ready, small model |
| L4 | 16 | 6 | — | — | 0 | — | pool with L3/L5 |
| L5 | 96 | 62 | 1.00 / 0.76 | 53% (40–63) | 3 | 3.7 | pool with L3 |

The gold set confirms that late-level approval rates below the literature's ~85–90% NDA→approval are real for this
population at these cut-offs (the corrected rates match the pipeline's), not a labelling artefact. P5 must treat the
label as a noisy measurement (sensitivity/specificity priors from `label_error.json`), pool L4/L5 with L3, and keep
late-level models to a handful of features.

2026-11-01 data fixes (each with a regression test; 02–05 rerun): parser (non-US filings naming an application
acronym, capitalised IND acceptances, "standard review" comparisons, "accepted the notion of"); 03 (a filing parsed for
several programmes stays with those whose indication its sentence names; filing events from multi-drug documents need
the drug in the sentence); 04 (an approval before the programme's own Phase 3 completion is withheld only if another
programme of the drug had already finished its Phase 3); 02 (`strip_dose` keeps code names: 'TAK-861' no longer
resolves to CDK9). Open items: `data/gold/findings.md` 21–24.

Deviation, decided 2026-11-01:
- **Per-level target Y_LEVEL** (`labels.level_target`, column `y_level` / `event_level` / `t_level_days` in
  landmark_labels and landmarks). L1–L2: Y_PROG (D1). From L3 on: approval within 3 years of the level's cut-off
  (83% of L3 and 96% of L5 eventual approvals arrive within 3 years), because the 7-year clock from the first Phase 3
  counted late filers as failures (41 of 107 L3 "failures" were approved after the window, e.g. ganaxolone,
  migalastat, pacritinib). Y_PROG is kept unchanged for reference and for the programme-level analysis.

**New features (2026-11-01, before P5):** all rule-based (D6), shared by training and live cases, each with tests.
- **Primary-endpoint p-values from readout press releases** (`press_release_parser.readout_pvalue`): the worst
  p-value in the readout sentence or the next 8 that say "primary", skipping secondary/subgroup/post-hoc
  sentences, other phases and the "About ..." boilerplate. 17/17 correct on review samples; found in ~13% of
  readout releases. Registry results still take priority. Phase 3 |z| coverage at L2+: 2–11% → 15–28%.
- **Phase 3 design:** double-blind, data monitoring committee, and the **endpoint class** of the registered primary
  outcome (`stats_transforms.endpoint_class`: survival, time-to-event, response, clinical event, biomarker,
  scale/score, PK/safety, other). Overall-survival endpoints have the *lowest* approval rate (26% at L1 vs 44%
  biomarker), the opposite of v1's endpoint weights (mostly oncology risk).
- **Sponsor's prior complete response letters** (press-release CRLs before the cut-off; the 2025 FDA CRL database
  stays label-only). Raw association is positive (experience proxy); adjusted for sponsor approvals it is null.
- **Era for live cases** (`anchor_year` from the earliest Phase 3 start).

**P5 production model (2026-11-01): `scripts/09_production_model.py`, `fdaengine/model/landmark.py`,
`models/landmark-v1/`, `reports/validation.html`.** Bayesian logistic model across L1–L5 (PyMC, sampled with
nutpie): level intercepts with weak literature priors, effects shared across levels with stage deviations (L1 | L2 |
L3–L5), a TA random effect, and the label as a noisy measurement (sensitivity/specificity Beta priors from
`label_error.json`). Predictions are P(true outcome). Missing inputs contribute nothing (training-mean imputation;
missingness is never a feature). 1,992 training rows; final fit R̂ ≤ 1.005, 0 divergences.

Cross-validated (5 folds grouped by drug; scores against the observed labels on the observed-label scale):

| Level | n | Brier | Best benchmark (Brier) | Provisional (app today) | Cal. slope (95% bootstrap) | AUROC | P5 target |
|---|---|---|---|---|---|---|---|
| L1 | 1,254 | 0.218 | 0.220 logistic, same features | 0.242 | 0.89 (0.68–1.08) | 0.65 | met |
| L2 | 345 | 0.218 | 0.221 | 0.278 | 0.93 (0.62–1.27) | 0.68 | met |
| L3 | 281 | 0.215 | 0.214 "endpoint met" alone | 0.261 | 0.62 (−0.12–1.41) | 0.57 | tie (Δ +0.001, CI −0.005 to +0.008) |
| L5 | 96 | 0.214 | 0.211 "endpoint met" alone | 0.288 | 1.21 (0.34–2.27) | 0.66 | tie (Δ +0.003, CI −0.011 to +0.019) |

Significantly better than the provisional model and the level base rate at every level except L3 (base rate tie);
best Brier score overall (0.218 vs 0.221–0.255). Mean predicted P(true approval) matches the gold-set corrected
rates (L1 43% vs 40%, L2 56% vs 51%, L3 70% vs 67%). After filing, little public pre-decision information
predicts the outcome beyond the base rate and whether the endpoints were met: L3/L5 forecasts are calibrated but
only weakly discriminating, and the report should say so. Strongest effects: Phase 3 met (OR 3.8 per step),
Phase 2 |z| (2.6/SD), ≥2 pivotal trials (2.3), large sponsor (1.9), sponsor approvals (1.3/SD); AdCom scheduled
0.56 at L3–L5.

Deviations, decided during P5:
- **Script numbering:** 08 is the data-readiness check and 09 the production model; payer analogs → 10, model card
  → 11, market backtest → 12.
- **Sampler: nutpie** (numba), because PyTensor's C backend needs g++, which this Windows machine lacks.
- **Late levels (L3–L5) use only pre-registered features** (`Spec.late`: Phase 2/3 met, Phase 3 |z|, large sponsor,
  ≥2 pivotal trials, priority review, accelerated pathway, AdCom scheduled, PDUFA extended; H7/H9/H11/H12/H15). With
  every feature the late levels overfit (CV calibration slope 0.16 at L3, −0.02 at L5).
- **No TA effect at L3–L5** (`TA_LATE = False`): the TA prior rate predicted worse than the level base rate there,
  and dropping TA improved the late-level CV Brier (L3 0.217 → 0.215, L5 0.222 → 0.214). This choice was informed
  by the validation results, so the late-level CV scores are slightly optimistic.
- **The app used the provisional model by default until P6d** (now the fitted model; `model/registry.py`); L6 falls
  back to the provisional model. Live cases do
  not yet compute sponsor prior approvals/CRLs, moiety-approved or TA, so those inputs count as missing (P6).
- Not built yet: Y_NEXT model, timing (competing-risks) model, label models (§7.5); recalibration was not needed.

**Data-source extensions (2026-11-01, after P5), each tested with a paired ablation** (identical rows, folds hashed
by drug, 3 fold seeds; changing the fold assignment alone moves CV Brier by up to ~0.006, so single runs are not
comparable across data builds):
- **PubMed trial publications** (`sources/pubmed.py`, `scripts/01b_pubmed.py`): 38,039 Phase 2/3 trials searched by
  NCT ID, 24,711 articles, 14,056 trials with a paper, 5,202 with a main-paper primary result (dated by electronic
  publication; ~85% precise on review). **Not used as features** (`features.build.USE_PUBMED = False`): far less
  predictive than registry results (L1 Phase 2 AUC 0.15–0.54 vs 0.81) and worse CV Brier at L1, L3 and L5 (overall
  +0.0010, every seed). Live cases still save the main paper's result sentence as a source file.
- **Federal Register AdCom meeting notices** (`sources/federal_register.py`, `scripts/01c_federal_register.py`):
  1,969 notices 1995–2026 → 748 ADCOM_SCHEDULED events dated by publication (meeting date as value). A drug must be
  named in an agenda sentence citing its application (name-only matching was ~45% precise; now ~65%: combination
  products and multi-item notices remain). Net effect null (L3 +0.0008, L5 −0.0009).
- **AdCom vote parsing widened** ('The Committee voted 21 to 5', 'AMDAC voted (7-6)', 'voted 11 to 8 against' read
  as 8 for): 42 → 61 votes; L4 training rows 16 → 23.
- **Not done:** sponsor investor-relations sites mostly refuse automated requests (bot protection; not
  circumvented); newswire archives (PR Newswire allows it in robots.txt) raise a terms-of-use question for the user;
  FDA's live site keeps AdCom minutes only from 2023 (older ones are in the FDA web archive, a separate job).
- **Conclusion:** the public pre-decision record available without newswire or archive scraping is close to
  exhausted for these fields; the model's late-level limits are a data-availability limit, not a modelling one.

**P6a timing model (2026-11-01): `scripts/10_timing_model.py`, `fdaengine/model/timing.py`, `models/timing-v1/`,
`reports/timing.html`.** P(approved by month h) = P(approved within the horizon) [landmark model] × G(h | approved).
G is estimated directly as a CDF: each approved row contributes "approved by h?" at a monthly grid up to its horizon,
fitted by one penalised logistic regression on stage-specific splines in log h, covariates × log h (large sponsor,
BTD, orphan, priority review, accelerated pathway, AdCom scheduled, PDUFA date known) and a "PDUFA date passed by h"
step (new shared feature `months_to_pdufa`, latest announced goal date, L3+; known for 210 late rows). Monotone in h;
50 drug-bootstrap refits give 90% intervals. Cross-validated (folds hashed by drug): **calibrated at every level and
horizon** (20 of 20 level × {6, 12, 24, 36}-month cells within the observed share's 95% interval; e.g. L3 12 months
53% predicted vs 55% observed). It beats the level's empirical curve where covariates matter (L3 6 months Brier 0.201
vs 0.218; L5 6–12 months) and ties it elsewhere: most timing information is the level itself. Cases with the fitted
model get `approval_timing` in `prediction.json` (6/12/24/36 months and 1 January of the next two plan years) and an
"Approval timing (payer view)" panel in the report. Check later: at L1 the median time from cut-off to approval
among approved programmes is 23 months, short for Phase 3 + review; the registry often misses a drug's earliest
Phase 3, which makes the anchor late.

**P6b new-indication labels (2026-11-01): `scripts/01d_approval_letters.py`, `labels.approval` rules 3b–5,
`data/gold/findings.md` items 25–33.** New uses of approved drugs are now attributed from each efficacy supplement's
(or new-use NDA's) **FDA approval letter** (4,762 letters; "provides for ..."), a post-decision document used as a
label only. Rules: a letter naming the indication (75% of keywords in one passage, programme indication + seed
trials) decides the supplement and its date, counting from the Phase 3 start; a new-use NDA (incl. type 2) whose
letter or label covers the indication competes on date; the label decides only for supplements without a letter; a
novel drug's later supplement needs its first approval to be clearly another use; Drugs@FDA abbreviations no longer
break sponsor matching. Gold set extended with **40 lifecycle programmes** (252 total, 75 lifecycle): lifecycle
Y_PROG **67.6% → 83.6%** (n=73, 95% ~75–92%; target 85%), novel unchanged at 89.5%, approval dates within 30 days
80% → 88%. The remaining errors are mostly upstream (trial grouping, drug resolution) or need line-of-therapy
judgement; more keyword rules would fit the gold set. Paired ablation: the fitted model is unchanged (all levels
within ±0.0007 Brier). Script 09 now uses folds hashed by drug and 3 repeats (single runs moved a level's Brier by
up to 0.017 between builds). Timing still calibrated in all cells.

**P6c label outcomes given approval (2026-11-01): `scripts/01e_approval_documents.py`,
`dataset/label_outcomes.py`, `scripts/11_label_models.py`, `model/label_model.py`, `models/label-v1/`,
`reports/label_outcomes.html`, `data/gold/findings.md` items 34–37.** Labels come from documents the FDA released at
approval (labels only, D2): the approving submission's **approval letter** and **label as approved** (Drugs@FDA
ApplicationDocs, 1,288 submissions; 92% of approved programmes have each) and **FDA's list of CDER accelerated
approvals** (350 entries; fda.gov/media/151146). Among 1,021 approved novel drugs: accelerated approval 13.0%,
boxed warning at approval 30.6%, REMS 12.0% from 2008 but 6.8% from 2012 (4.8% with elements to assure safe use).
Parser checks: the accelerated approval list beat the letter wording in all 12 disagreements; the label as approved
beat today's openFDA label (boxes are added and removed after approval); REMS 20/20 correct on a fresh sample.

Model: one penalised logistic regression per outcome on 989 approved landmark rows (596 programmes, 485 drugs,
L1–L5; an open 7-year window is fine because the outcome is known once approved), stage intercepts
(L1 | L2 | L3–L5), the landmark inputs cut off at the level, era as cumulative steps of the Phase 3 start year, penalty
chosen by grouped cross-validation, 50 drug-bootstrap refits for 90% intervals. Validated with folds hashed by drug
(3 repeats) and a temporal holdout (latest 20% of cut-offs):

| Outcome given approval | Stage | n (events) | Brier | vs stage base rate (Δ, 95% CI) | vs TA base rate | AUROC | Cal. slope | Temporal holdout (model vs stage rate) | Shown as |
|---|---|---|---|---|---|---|---|---|---|
| Accelerated approval | L3–L5 | 244 (42) | 0.112 | −0.030 (−0.051 to −0.012) | −0.013 (−0.033 to +0.005) | 0.83 | 1.09 | 0.152 vs 0.186 | drug-specific |
| Accelerated approval | L1 / L2 | 506 / 201 | 0.064 / 0.062 | +0.002 / +0.003 (ties) | ties | 0.66 / 0.72 | 0.51 / 0.61 | 0.147 vs 0.157 / 0.144 vs 0.156 | typical share |
| Boxed warning | L1 / L3–L5 | 499 / 240 | 0.213 / 0.205 | −0.012 / −0.016 (CIs exclude 0) | −0.013 / −0.018 | 0.62 / 0.61 | 0.96 / 0.95 | 0.196 vs 0.194 / 0.206 vs 0.194 (worse) | typical share |
| REMS (approvals from 2012) | all | 765 (51) | 0.061 | −0.002 (−0.006 to +0.003) | −0.002 | 0.68 | 0.81 | 0.058 vs 0.061 | typical share |

- **Accelerated approval is predictable once the filing is known** (reported accelerated pathway OR ≈ 5, oncology
  OR ≈ 2.6, progression-free / survival endpoints). It is conservative at the extremes (oncology drugs on a reported
  accelerated pathway: 9 of 9 got it; the model says 50–80%) and under-predicts the recent rise (temporal holdout
  15% predicted vs 23% observed after filing). Before filing it is no better than the typical share.
- **Boxed warning: weak.** Cross-validation beats both base rates (psychiatry, combinations, larger trials, earlier
  eras raise it; AUROC 0.61), but it does not beat the base rate on the most recent programmes (and predicts 19%
  where 25% were boxed), so reports show the typical share.
- **REMS: no skill beyond the typical share.** Its apparent skill on 2008–2012 approvals was the Medication-Guide-REMS
  era (39% of approvals then). A linear era term extrapolated it to a 1% REMS rate in the temporal holdout; era is now
  stepwise and REMS is trained on approvals from 2012.
- **Label narrower than the trial population: not built.** The rule-based label (approved indication restricted by
  prior therapy, no alternatives, a companion test, severity or age, against the registered inclusion criteria of
  Phase 3s completed before approval) was right for 13 of 25 flagged programmes on a fresh sample; half the errors
  are upstream trial grouping and approval attribution (findings 31, 37). The column stays in `label_outcomes.parquet`.

`prediction.json` gets `label_outcomes` (fitted model only) and the report an "Expected Label If Approved (payer view)"
panel: the drug-specific probability where the model beat the stage's typical share in both cross-validation and the
temporal holdout (today: accelerated approval after filing), otherwise the typical share with the model's figure for
reference. Novel drugs only; lifecycle (new-indication) programmes inherit most label features from the drug's existing
label and are not modelled yet.

Deviations, decided during P6c:
- **Script numbering:** 11 is the label models; model card → 12, market backtest → 13.
- **Separate penalised logistic models with bootstrap intervals** instead of PLAN §7.5's multivariate Bayesian model
  with a shared random effect (as for timing: simpler, validated, and each outcome is shown on its own).
- **Narrow label dropped** from the outputs (measured precision ~50%).
- **Rebuild order:** 03 → 04 → 05 → gold_build → gold_score → 06 → 07 → 08 → 09 → 10 → 01e → 11. Script 01e reads
  script 04's approvals despite its number (downloads are cached, so a rerun only re-reads the PDFs; the first run took
  12 minutes).

**P6d live-case context (2026-11-01): `features/context.py`, `features/build.py`, `predict/path.py`,
`tests/test_context.py`.** Live cases now compute the context inputs the fitted models were trained on, with the
training definitions (script 05 imports the same `CompanyDates`, sponsor approval/CRL tables and modality rule;
recomputing modality and both sponsor counts for all 10,584 landmark rows reproduces the stored values exactly):
therapeutic area (each Phase 3's primary MeSH condition through the leaf → ancestor map learned from all trials,
replacing Claude's free-text area), modality, active ingredient already approved, sponsor's prior approvals and
announced CRLs, combination product. Source `reference`, each with a detail line; the probability path recomputes
them at each level's cut-off. On 300 random training rows run through the live code: sponsor counts, modality and
combination 100% identical, TA 99% (live cases also use linked Phase 3s), moiety approved 98.5% (4 salt/ester display
names resolve to the parent ingredient). The reference tables load once per process (~17 s).
- **A case's own approval is never "already approved".** The case's own applications are excluded (bepirovirsen's
  NDA 221276 was approved on 2026-10-20, nine days before its live L3 cut-off). When the own application is approved by
  the cut-off, a data flag says the outcome is already known.
- At L3–L5 the landmark model uses none of these inputs (by design, P5), so filed-stage probabilities do not move;
  L1–L2 probabilities and the label panel (therapeutic area) do.
- Found, not fixed: the live timeline attributed momelotinib's Phase 3 news from GSK's multi-drug pipeline releases
  to bepirovirsen (a "Phase 3 not met" event), which sets its L3 forecast to 54%. The press-release parser's drug
  proximity rule fails on pipeline tables.
- **The app now uses the fitted model by default** (user decision 2026-11-01): landmark-v1 + timing-v1 + label-v1.
  `FDAENGINE_MODEL=provisional` restores the prior-only model; L6 still falls back to it.

**Next (payer first, D9):** P9 payer module → P10 hardening.
The investor module (P11) is paused until the user asks for it.
---

## 0. What's wrong with v1 (and what v2 fixes)

| v1 behaviour | Problem | v2 replacement |
|---|---|---|
| Hand-picked weights and penalty curves | Never fitted to data | Coefficients and curve shapes estimated from historical programmes; each v1 assumption tested as a hypothesis |
| One snapshot, one probability | Can't tell you how the outlook changes as news arrives | Forecast at any selected milestone + a probability path across milestones |
| `TA_BASE_RATES`, `PHASE_RATES` | Mix different transition rates | Hierarchical base rates per level, with literature priors |
| "95% CI" = `0.09 + 0.08*(…)` | Has no statistical meaning | Bayesian posterior credible interval |
| Hardcoded feature importances | Fake | Per-prediction log-odds contributions + bootstrap global importance |
| Copy a prompt to an LLM, paste the JSON back | Manual, no provenance | Automated download + headless Claude extraction; every field cites document, page and quote |
| Only a yes/no probability | Insurers need *when* and *what label*, not just *whether* | Time-to-approval distribution, label-characteristic predictions, budget impact |
| No market context | Can't tell whether there's an edge | Event-study backtest against actual stock reactions (built last) |

The v1 **visual design** (paper/ink palette, Playfair + DM Mono, result header, flags, contribution bars) is kept as
the report template.

---

## 1. System overview

```
┌──────────────────────────────┐
│ Streamlit UI (app/)          │  drug, indication, LEVEL selector, mode: live | backtest | scenario, AUDIENCE: payer | investor | neutral
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐  1. RESOLVE   names/code names → trials, applications, sponsor, ticker
│ Python case pipeline         │  2. TIMELINE  build the programme's event timeline → cut-off date for the chosen level
│ (fdaengine/)                 │  3. ACQUIRE   download everything dated ≤ cut-off → cases/<slug>/raw/ (+ manifest)
│                              │  4. PREPARE   PDF → page-tagged text + tables; ClinicalTrials.gov results → JSON
│                              │  5. CLAUDE    `claude -p` + prompts/extract_case.md → extracted/fields.json
│                              │  6. VALIDATE  schema, verbatim-quote check, statistical consistency, reconcile
│                              │  7. FEATURES  the same feature builder used for training, for that level
│                              │  8. PREDICT   level-specific Bayesian model → posterior probabilities
│                              │  9. EXPLAIN   contributions, flags, comparables, sensitivity, OOD, path
│                              │ 10. APPLY     payer module: approval-by-date, label, analog pricing, budget impact
│                              │               investor module (last): ticker, catalyst, reaction sizes, market-implied p
│                              │ 11. REPORT    report.html / report.pdf (+ memo from a second `claude -p` run)
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Trained models (models/vX/)  │  built offline by scripts/01…11: one landmark model across levels,
│                              │  timing + label models, model card with hypothesis tests, calibration per level
└──────────────────────────────┘
```

The offline (training) and online (case) pipelines **share the same parsers and feature code**, so the model is
trained on exactly the kind of inputs it will see in use.

---

## 2. Forecast levels (landmarks)

Each level is a **landmark**: a point in a programme's life with a defined information cut-off. You choose the level
in the UI.

| Level | Name | Cut-off (information up to…) | New information available at this level |
|---|---|---|---|
| **L1** | Phase 2 complete | Day before the first Phase 3 topline result | Phase 1/2 results, mechanism, genetics, sponsor, orphan/FT/BTD so far, Phase 3 *design* as registered |
| **L2** | Phase 3 readout | Day after the (last) pivotal Phase 3 topline release | Phase 3 efficacy (met/missed, z-stat, effect size), Phase 3 safety, number of positive pivotal trials, secondaries |
| **L3** | Filed | Day after the FDA accepts the filing (the company announces it) | Submission date, priority vs standard review, PDUFA date, BTD/accelerated pathway, rolling submission, time from readout to filing, refuse-to-file history |
| **L4** | Post-AdCom | Day after the AdCom vote (if one is scheduled) or mid-cycle | AdCom scheduled at all?, briefing-document tone, vote %, question wording |
| **L5** | Pre-PDUFA | 7 days before the PDUFA date (after any extension) | PDUFA extension (major amendment), inspection findings published (Form 483s), REMS/label discussions disclosed, late-cycle disclosures |
| **L6** | Resubmission after CRL | Day after the resubmission is accepted | CRL contents (FDA now publishes CRLs), resubmission class (1 or 2), what was fixed, new data |
| **Auto** | Latest available | Today (live) or a chosen date (backtest) | The most advanced level the programme has reached |

**Behaviour in the UI:**

- **A level the drug has already passed** (e.g. it's filed and you pick L1): the engine **restricts inputs to that
  level's cut-off**. This shows what the model would have said back then, which is useful for backtesting and for
  seeing how much each milestone moved the probability.
- **A level the drug hasn't reached yet**: greyed out, unless you switch to **scenario mode** and supply assumed
  values, e.g. "Phase 3 meets its endpoint with HR 0.75 (0.62–0.90)". The engine then returns a conditional forecast.
- **Probability path:** a single click runs every reached level and plots P(approval) through L1 → L2 → … with the
  event that caused each change.

---

## 3. Prediction targets

| Target | Definition | Available at |
|---|---|---|
| **Y_PROG (headline)** | The programme (drug × indication) gets FDA approval within 7 years of Phase 3 start (resubmissions count) | All levels |
| **Y_NEXT** | The FDA approves at the next action date (vs CRL) | L3–L6 |
| **T_APP (timing)** | Time from the cut-off to approval, with failure as a competing event → **P(approved by date d)** for any d (e.g. 1 January of the next plan year) | All levels |
| **Label outcomes** | Given approval: boxed warning? REMS? Accelerated (vs traditional) approval? Label population narrower than the pivotal-trial population? | All levels (most informative at L2+) |
| Y_P3 (component) | Pivotal Phase 3 meets its primary endpoint | L1 only (shown as a component) |

Label outcomes are *training labels* taken from the eventual FDA label. Using post-decision documents as **labels**
is fine; they are never used as **inputs**.

At L1 the headline is decomposed as:

```
P(approval) = P(Phase 3 success) × P(approval | Phase 3 success)
```

**Censoring:** recent programmes without an outcome are handled with inverse-probability-of-censoring weights and
**competing-risks** analysis (approval vs failure/termination: Aalen–Johansen, Fine–Gray), plus a fully-resolved-only
sensitivity analysis. The competing-risks model is also the **timing model** behind T_APP: its cumulative incidence
curve *is* P(approved by date d).

**Cut-off enforcement (leakage):** every source has a `source_date`. The feature builder **refuses** anything after
the level's cut-off, and unit tests check this. ClinicalTrials.gov record history is used to rebuild registrations
as they stood at the cut-off. FDA review packages, approval letters and final labels are **never** inputs, because
they appear after the decision.

---

## 4. Data sources

| Source | Provides | Access | Levels |
|---|---|---|---|
| **AACT** (daily PostgreSQL mirror of ClinicalTrials.gov) | Design, arms, dates, **structured outcome analyses (p-values, estimates, CIs)**, adverse events per arm, participant flow | Free account or flat files | L1–L2 backbone for training |
| **ClinicalTrials.gov API v2** + record history | Per-trial JSON, historical versions | Free | Live cases; cut-off reconstruction |
| **SEC EDGAR full-text search + filings** | Press releases (8-K Ex-99): toplines, filing acceptance, priority review, **PDUFA dates**, extensions, AdCom news, CRLs, label/REMS disclosures; 10-K pipeline tables; sponsor CIK ↔ ticker | Free (User-Agent required) | **Event timeline for L2–L6**, failure labels |
| **Drugs@FDA bulk download** | Applications, action dates, outcomes, marketing status (Rx/OTC) | Free zip | **Outcome labels + decision dates** (not inputs) |
| **openFDA** `drug/label`, `drug/drugsfda` | Pharmacologic class / mechanism of previously approved drugs; indications | Free API | Mechanism precedent; indication matching |
| **FDA Advisory Committee calendar & materials** | Meeting dates, briefing documents (released ~2 business days before), questions, votes, minutes | Scrape fda.gov | L4 |
| **FDA CRL database** (published from 2025, including historical CRLs) | CRL text and reasons | Free | L6 inputs; CRL-reason taxonomy for research |
| **FDA Data Dashboard: inspections, 483s, warning letters** | Site compliance history, **using its publication date** | Free | L3–L5 (CMC risk) |
| **FDA Orphan Drug Designations** | Designations with dates | Free | All |
| **PubMed / Europe PMC** | Phase 2/3 publications (effects, CIs, dose-response) | Free API | L1–L2 |
| **Open Targets Platform** | Genetic evidence for the target–disease link | Free GraphQL | All |
| **RxNorm / UNII (GSRS) / MeSH** | Drug and condition normalisation | Free | Entity resolution |
| **CMS Medicare Part D & Part B drug spending dashboards** | Annual spend, claims, beneficiaries and spend per beneficiary by drug | Free | **Payer module:** analog prices and launch uptake curves |
| **Medicaid State Drug Utilization Data** | Quarterly prescriptions/reimbursement by NDC | Free | Payer module: quarterly uptake curves for analogs |
| **CMS ASP pricing files / NADAC** | Part B average sales price; pharmacy acquisition cost | Free | Payer module: price benchmarks |
| **Disease prevalence** (CDC, NIH/NCI SEER, GBD, orphan prevalence from Orphanet) | Eligible-population estimates | Free | Payer module: budget impact |
| **ICER reports** | Independent value-based price benchmarks for many new drugs | Free (registration) | Payer module: price range check |
| *Optional, paid:* WAC list prices (First Databank, Medi-Span), claims data (IQVIA) | Exact list prices, real utilisation | Paid / customer-supplied | Payer module; an insurer customer would bring its own claims data |
| **Market data** (Alpaca: you already have credentials from APEX; or yfinance) | Daily prices for sponsor stocks + XBI/IBB benchmarks | Free tier (check history depth; delisted tickers are a gap) | Investor module (built last) |
| *Optional, paid:* historical options data (ORATS, Polygon, Cboe) | Implied move into PDUFA/AdCom dates | Paid | Investor module: market-implied probability |
| *Optional, paid:* Citeline/Biomedtracker | Cleaner failure labels, event dates | Paid | Labels |
| **Literature priors** | Hay et al. 2014 (*Nat Biotechnol*); Wong, Siah & Lo 2019 (*Biostatistics*); BIO/Informa/QLS success-rate reports; Minikel et al. 2024 (*Nature*, genetic support); O'Hagan et al. 2005 (assurance) | Papers | Priors + external benchmarks |

---

## 5. Offline research pipeline: the historical dataset

Numbered, idempotent, cached scripts in `scripts/`. **No LLM calls anywhere in this pipeline** (D6).

| Script | Does | Output |
|---|---|---|
| `01_download_bulk.py` | AACT, Drugs@FDA, openFDA, orphan DB, CRL DB, inspection data, AdCom pages, Open Targets slices | `data/raw/` + sha256 manifest |
| `02_build_programmes.py` | Group industry Phase 3 trials into drug × indication programmes (RxNorm/UNII + MeSH); link the earlier Phase 1/2 trials | `programmes.parquet` |
| `03_build_timelines.py` | **Event timeline per programme**: P3 topline, submission, acceptance, review type, PDUFA date, extensions, AdCom, CRL, resubmission, decision. Sources: EDGAR press releases parsed by **rule-based extractors** (phrase patterns + date parsing), FDA AdCom calendar, Drugs@FDA action dates | `events.parquet` (event, date, source, parser confidence) |
| `04_label_outcomes.py` | Y_PROG, Y_NEXT per action, Y_P3, T_APP; label outcomes from the eventual openFDA label (boxed warning, REMS, accelerated approval, indication-text breadth vs trial eligibility); censoring status | `labels.parquet` (`label_source`, `label_confidence`) |
| `05_build_landmark_rows.py` | One row per programme **per level it reached**, with features computed strictly from information ≤ that level's cut-off | `landmarks.parquet` |
| `06_eda_and_univariate.py` | Missingness audit, distributions, univariate tests + FDR, per level | `reports/eda.html` |
| `07_fit_models.py` | §7 models | `models/vX/` |
| `08_data_readiness.py` | Label error from the gold set, statistical capacity per level (built 2026-11-01) | `reports/data_readiness.html` |
| `09_production_model.py` | P5: fit, temporal + grouped CV validation, calibration, benchmarks | `models/landmark-v1/`, `reports/validation.html` |
| `10_timing_model.py` | P6a: approval timing model, validation | `models/timing-v1/`, `reports/timing.html` |
| `11_payer_analogs.py` | Analog launch library: approved drugs × CMS Part D/B + Medicaid spend/utilisation by year since launch; Bass-diffusion uptake fits; price per patient-year by class/indication (§8.1) | `data/processed/analogs.parquet` |
| `12_model_card.py` | Dataset, hypothesis verdicts, effect sizes, performance per level, timing and label model accuracy | `models/vX/model_card.html` |
| `13_market_backtest.py` | **Paused (D9):** event study + strategy backtest (§8.2) | `reports/market_backtest.html` |

**Optional, manual only:** `scripts/claude_sample_extract.py --n 50` runs the case-extraction prompt over a sample of
historical programmes, one `claude -p` call at a time, whenever **you** decide to run it. It's used for (a) checking
the rule-based parsers against Claude on a gold set and (b) testing text-only features (§6.3). The pipeline never
depends on it.

**Gold set:** about 150 programmes, stratified by TA, era, level and outcome, hand-checked for programme grouping,
event dates and labels. Parser accuracy is reported in the model card.

**Expected size:** several thousand L1 rows; L3–L6 rows are limited to programmes with press-release timelines
(mostly public sponsors), roughly 1,000–2,000 applications. Shrinkage priors and pre-registered hypotheses keep
the number of parameters within what the event counts support.

---

## 6. Features by level

Each level inherits all features from earlier levels and adds its own.

### 6.1 Feature table

| Level | Family | Features |
|---|---|---|
| L1 | **Phase 2 efficacy** | Primary endpoint met; **z-statistic** on a common scale (from HR/OR/mean difference + CI or p); standardised effect → log-OR (Chinn 2000); random-effects pooled Phase 2 effect + I² across trials |
| L1 | **Phase 2 design** | Randomised / placebo-controlled / blinded / single-arm; N; number of doses; **dose-response trend test** |
| L1 | **Phase 1/2 safety** | SAE relative risk with CI; death risk difference (Newcombe CI, Fisher exact p); AE-discontinuation rate; hepatic/QT/CV signals from MedDRA terms |
| L1 | **Phase 3 design (registered)** | Endpoint class, comparator, superiority vs non-inferiority, blinding, planned N, number of pivotal trials |
| L1 | **Bridge** | **Bayesian assurance** (predictive power of Phase 3 from the Phase 2 effect, shrunk for the winner's curse); endpoint concordance between Phase 2 and Phase 3 |
| All | **Drug / mechanism** | Modality; first-in-class vs validated mechanism; repurposed (already approved elsewhere); genetic support (Open Targets) |
| All | **Indication** | TA; historical success rate in the indication (computed only from programmes resolved before the cut-off); approved standard of care exists; rare disease |
| All | **Sponsor** | Prior approvals, prior success rate, size (market cap / private), big-pharma partner, first-ever NDA/BLA |
| All | **Context** | Era (year spline); Rx; NME vs biologic vs supplement |
| L2 | **Phase 3 efficacy** | Met / missed / mixed; **z-statistic**; effect + CI; consistency across pivotal trials (meta-analysis, I²); key secondaries met (beta-binomial shrunk); **Phase 3 effect vs Phase 2 effect** (shrinkage observed) |
| L2 | **Phase 3 safety** | SAE RR with CI; death imbalance (Newcombe/Fisher); discontinuation; boxed-warning-class signals; **v1's asymmetric penalties re-tested as splines** |
| L2 | **Readout signals** | Single pivotal vs two; subgroup-driven claims; endpoint changed from protocol; "clinically meaningful" language without significance |
| L3 | **Filing** | Priority review; accelerated approval pathway; BTD; rolling submission; RTOR; **months from readout to filing**; refuse-to-file history; standard/priority PDUFA clock |
| L4 | **AdCom** | Scheduled at all (itself a signal); vote % on efficacy and benefit–risk questions; question framing; briefing-document FDA-reviewer concerns (rule-based phrase extraction) |
| L5 | **Late cycle** | PDUFA extended (major amendment); 483s/OAI inspections at listed sites published before the cut-off; REMS or label discussions disclosed; delay announcements |
| L6 | **Resubmission** | CRL reason category (clinical efficacy / safety / CMC / labelling / other); Class 1 vs 2 resubmission; new trial required?; months since CRL |
| All | **Missingness** | Phase 2 results posted? Phase 3 results posted? Published? Each one tested as a feature |

### 6.2 Missing data

Little's MCAR test plus a logistic model of missingness on observed covariates. **MICE** (m = 20, Rubin's rules) for
frequentist models; joint imputation in the Bayesian model where feasible; missingness indicators where absence is
informative. In live cases, every imputed field becomes a data-quality flag.

### 6.3 Text-only features

Some features have no structured source: post-hoc subgroup rationale, endpoint switches, spin, and the tone of FDA
concerns in the AdCom briefing. In the live case, Claude extracts these.

They enter the **model** only if the rule-based parsers can approximate them across the training set, or if you have
run the optional sample script on enough programmes to test them. Otherwise they appear in the report as
**qualitative analyst flags**, clearly labelled "not in model".

---

## 7. Statistical inference & modelling

### 7.1 Pre-registered hypotheses

Each gets an effect estimate with a CI, a test, and a verdict (supported / not / insufficient), reported per level:

| # | Level | Hypothesis |
|---|---|---|
| H1 | L1 | Phase 2 meeting its endpoint predicts approval; the **z-stat** adds information beyond met/not-met |
| H2 | L1 | **Winner's curse:** Phase 2 effects overestimate Phase 3 effects; estimate the shrinkage factor |
| H3 | L1 | Randomised Phase 2 beats single-arm; dose-response evidence helps; endpoint concordance helps |
| H4 | L1 | Bayesian assurance is predictive **and calibrated** on its own |
| H5 | All | Genetic support raises success; first-in-class lowers it; repurposing raises it |
| H6 | L2 | Phase 3 z-stat has a nonlinear effect beyond met/not-met (borderline p is penalised) |
| H7 | L2 | Two positive pivotal trials beat one; inconsistency between trials (I²) hurts |
| H8 | L2 | **SAE and death imbalances have convex (asymmetric) penalties**: v1's assumption, tested with splines and segmented regression |
| H9 | L3 | Priority review / BTD predict approval independently of efficacy |
| H10 | L3 | A long readout-to-filing delay signals problems |
| H11 | L4 | **AdCom effect is asymmetric** (negative votes hurt more than positive votes help); the act of scheduling an AdCom is itself negative |
| H12 | L5 | A PDUFA extension changes the odds (direction tested, not assumed) |
| H13 | L5 | Published manufacturing-site 483s/OAI findings raise CRL risk |
| H14 | L6 | CRL reason category (CMC vs efficacy) determines resubmission success |
| H15 | All | Sponsor size/experience matters; TA and review-division effects persist after controls |
| H16 | All | Era effects / regime shifts |
| H17 | All | Missing results (not posted) signal higher risk |
| H18 | L2+ | Label restrictions (boxed warning, REMS, narrow label) are predictable from pre-decision safety and design data |
| H19 | All | Time to approval is shortened by priority review / BTD / RTOR (hazard ratios from the timing model) |
| I1 | All | Information value: how much of the final uncertainty each milestone resolves |
| M1 | Market (last) | Model probability before an event predicts the stock's abnormal return at that event, i.e. the market hasn't fully priced the model's information (§8.2) |

### 7.2 Univariate screening (exploratory)

Fisher's exact / χ² with ORs and Wilson CIs; Mann–Whitney U; univariate spline logistic regression;
**Benjamini–Hochberg FDR at q = 0.05**; forest plots per level; correlation/VIF clustering. This stage is exploratory
only.

### 7.3 Multivariable frequentist models (statsmodels)

- Logistic regression with restricted cubic splines; **likelihood-ratio tests** per variable; profile-likelihood CIs.
- **Firth** penalised logistic regression for near-separation (e.g. "Phase 3 missed" ≈ never approved).
- **GEE / mixed-effects** with clustering by programme (one programme contributes several landmark rows), drug and
  sponsor.
- Pre-specified interactions only (e.g. single-arm Phase 2 × oncology; AdCom vote × priority review).
- **Competing risks** (Fine–Gray) for approval vs failure, which also gives time-to-decision.
- Diagnostics: influence, VIF, linearity of the logit, Hosmer–Lemeshow (supplementary).

### 7.4 Variable selection with stability

Elastic net (cross-validated); **stability selection** (500 subsamples, keep features with selection probability
≥ 0.7); bootstrap inclusion frequencies; permutation tests.

### 7.5 Production model: Bayesian hierarchical landmark model (PyMC + ArviZ)

One model across all levels. **Coefficients are partially pooled across levels**, so a feature's effect can change
from L1 to L5 but borrows strength where the data are thin:

```
logit P(Y_PROG = 1 | row i at level ℓ) = α_ℓ + α_TA[ta] + α_ind[ind] + α_sponsor[s] + u_programme
                                          + Σ_j β_{j,ℓ} x_ij + f_ℓ(z_i) + g_ℓ(safety_i) + h(AdCom_i)

β_{j,ℓ} ~ Normal(β_j, τ_j)            # level-specific effects shrunk towards a shared effect
β_j     ~ regularised horseshoe       # sparsity: irrelevant features shrink towards 0
α_ℓ     ~ informative priors from the literature (e.g. Phase 3 → approval ≈ 50–60%; NDA → approval ≈ 85–90%)
f, g, h ~ penalised splines           # learned shapes, which replace v1's hand-drawn curves
```

- A parallel model for **Y_NEXT** (next-action approval) at L3–L6.
- **Timing model:** a Bayesian competing-risks / discrete-time hazard model (approval vs failure, monthly intervals)
  with the same feature set, giving posterior P(approved by date d) curves. It is cross-checked against the
  frequentist Fine–Gray model. PDUFA dates, once known (L3+), anchor the timing directly.
- **Label models:** multivariate logistic models (boxed warning, REMS, accelerated approval, narrow label),
  conditional on approval, with correlated outcomes handled through a shared random effect.
- Prior predictive checks; NUTS with 4 chains; R̂ < 1.01, ESS, divergences; posterior predictive checks per level and TA.
- **PSIS-LOO** comparisons of nested models: the Bayesian evidence for each hypothesis alongside the frequentist LRTs.
- **Per-prediction output:** posterior draws → median, 50% and 95% credible intervals, at the selected level (and at
  every reached level for the probability path).

### 7.6 Challenger: gradient boosting + SHAP

LightGBM per level (monotone constraints where theory is clear), SHAP for importance and interactions, compared with
the Bayesian spline shapes. Promoted only if it beats the Bayesian model on held-out **Brier score and calibration**.
Otherwise it's a cross-check, and big disagreements raise a flag.

### 7.7 Validation

- **Temporal split** (train on older programmes, test on more recent resolved ones) + rolling-origin backtests by year.
- **Per level:** Brier score + Brier skill score, log loss, AUROC, AUPRC, **calibration slope/intercept**, reliability
  diagrams, recalibration on the validation fold only if needed.
- **Information value per level:** log-loss reduction from L1 → L2 → … → L5, i.e. how much each milestone resolves.
- **Timing accuracy:** time-dependent Brier score / integrated Brier score and calibration of P(approved by date d)
  at 6, 12, 24 and 36 months. **Label models:** Brier score and calibration per label outcome.
- Calibration matters more than discrimination here: payers multiply probabilities by costs, so a miscalibrated
  probability gives a wrong budget even if the drugs are ranked correctly.
- **Benchmarks to beat:** TA base rate; literature phase-transition rates; assurance alone (L1); "Phase 3 met" alone
  (L2); **v1's formula** (L4/L5, where it has its inputs).
- Subgroups: Rx-only, oncology vs other, small vs large sponsor, NME vs supplement.

---

## 8. Application modules

The core engine (§2–§7) answers *whether*, *when* and *with what label*. Modules turn those answers into
decisions for a specific customer. They consume the core's posterior draws, so uncertainty carries through to their
outputs. **Build order: payer module first, investor module last.**

### 8.1 Payer / health-insurer module

**The customer's problem:** an insurer sets premiums and budgets 12–18 months ahead (e.g. rate filings for the next
plan year). Expensive new drugs (gene therapies, GLP-1s, specialty oncology) can blow a budget if they launch
unexpectedly, or tie up reserves if they don't. The insurer needs **which pipeline drugs will reach its members, when,
for whom, and at what cost**, with honest uncertainty.

**Outputs per drug:**

| Output | From |
|---|---|
| **P(approved by the start of plan year Y)** and the full approval-date distribution | Timing model (§7.5) |
| **Expected label:** P(boxed warning), P(REMS), P(accelerated approval), P(label narrower than trial population) | Label models (§7.5) |
| **Eligible population** (per 100k members, adjustable to the insurer's own member mix) | Prevalence sources × trial eligibility criteria (from ClinicalTrials.gov) × label-breadth probability |
| **Price benchmark:** expected annual cost per treated patient, as a range | **Analog pricing**: approved drugs in the same class/indication/modality, using CMS Part D/B spend per beneficiary, ASP/NADAC and ICER benchmarks; adjusted for orphan status and modality; shown as a distribution, not a point |
| **Uptake curve:** expected treated patients by quarter after launch | **Bass diffusion** model fitted to analog launches (CMS Part D annual + Medicaid quarterly utilisation), with hierarchical pooling by class |
| **Budget impact** (expected spend + 90% interval, per member per month) | Monte Carlo: P(approval by date) × label/eligible population × uptake × price, run for every posterior draw (ISPOR budget-impact good-practice structure) |

**Pipeline / portfolio view** (the feature an insurer would actually buy):
- Load a list of pipeline drugs (or a whole therapeutic area). The module runs the engine at "Auto" level for each one
  and aggregates by Monte Carlo into **total expected new-drug spend for plan year Y, with an interval**. It also
  gives the drugs that contribute most to the uncertainty, and a calendar of upcoming decision dates.
- Correlation between drugs: drugs in the same class share the class-level random effect (e.g. a class safety scare
  hits them all), so the aggregate interval isn't falsely narrow.
- **Watch triggers:** when a new event (topline, AdCom, PDUFA extension) moves a drug's probability by more than a
  set amount, it's flagged for re-budgeting.

**Validation (payer-specific):**
- Backtest a past plan year: freeze the engine at, say, July 2021 and forecast the 2022 new-drug approvals and spend
  for a sample therapeutic area. Then compare with the actual approvals and CMS spend.
- Metrics:
  - calibration of P(approved by date);
  - coverage of the 90% budget interval;
  - error of analog prices against actual launch prices (from later CMS data).

**Scope notes:**
- Free CMS data gives Medicare/Medicaid-based benchmarks. A commercial insurer customer would bring its own claims
  and price data, so the module accepts a customer data file (member mix, negotiated discounts, claims history) as an
  input rather than requiring it.
- Natural extensions later: biosimilar/generic entry timing (Orange Book/Purple Book exclusivity + patents), and
  label-expansion (supplement) forecasts for drugs already on formulary.

### 8.2 Investor module (paused, D9: not built until the user asks)

This is the **final phase of the build** (D8). It answers one question: **does the model know something before the
market prices it in?**

#### 8.2.1 Event study (historical)

- Sponsor → CIK → ticker (SEC `company_tickers.json` + historical name changes). Focus on small/mid-cap sponsors,
  where a single drug moves the stock.
- Events: Phase 3 topline, filing acceptance, AdCom, PDUFA decision, CRL.
- Market model with XBI/IBB as the benchmark: estimation window [−250, −30] trading days; **cumulative abnormal
  return (CAR)** over [−1, +1] (press-release timing handled: an after-close release → next session).
- Output: typical stock reactions to approval vs CRL, by market cap, TA and level. These feed the report's "expected
  move" panel.

#### 8.2.2 Does the model have an edge? (M1)

- Regress the event CAR on the model's pre-event probability, controlling for market cap, pre-event run-up and level:
  `CAR_i = a + b·p_model,i + c·controls + e`. **b > 0 (significant)** means the market had not fully priced the
  model's information.
- Surprise decomposition: CAR versus (outcome − p_model). Is the market's surprise consistent with the model's
  calibration?
- **Strategy backtest:** for example, long the sponsor N days before the PDUFA date when p_model is high and the stock
  hasn't run up; avoid or hedge when it's low. Includes transaction costs and borrow constraints. Outputs: hit rate,
  average return, Sharpe with **bootstrap CIs**, and the **deflated Sharpe ratio** (corrects for having tried several
  strategy variants).
- Biases handled: delisted tickers (survivorship), look-ahead in event dates, overlapping events.

#### 8.2.3 Market-implied probability (optional, needs options data)

With up/down jump sizes from §8.2.1 and the options-implied move into the catalyst:
`p_market ≈ (S₀ − S_down) / (S_up − S_down)`. The report then shows **edge = p_model − p_market**. Without options
data, the panel shows the model probability next to historical reaction sizes only.

#### 8.2.4 Optional APEX hook

Write `exports/catalysts.json` (drug, ticker, catalyst date, p_model, CI, level) in a format APEX's catalyst-day
review can read.

---

## 9. The headless Claude Code runs (APEX pattern, case analysis only)

### 9.1 Prompts

The prompts are files in `prompts/` (the equivalent of APEX's `REVIEWS.md`). The project `CLAUDE.md` holds the
standing rules:
- cite every field;
- never use sources dated after the cut-off;
- write only inside the case folder;
- never produce a probability.

| Prompt | Runs | Reads | Writes |
|---|---|---|---|
| `extract_case.md` | After acquisition | `cases/<slug>/text/`, `tables/`, `raw/ctgov/*.json`, `schemas/fields.schema.json`, `case.json` (level + cut-off) | `extracted/fields.json`: value, unit, source file, page, verbatim quote, confidence per field (including the event timeline) |
| `analyst_review.md` | After extraction | Same + `fields.json` | `extracted/analyst_flags.json`, `analysis/analyst_notes.md` (post-hoc subgroups, endpoint switches, spin, FDA concerns in briefing docs) |
| `write_memo.md` | After prediction | `prediction.json`, `payer.json` / `market.json` (per audience), `fields.json`, `analyst_flags.json` | `report/memo.md`, written for the selected audience (payer budget memo or investment memo), which may only use numbers from the JSON inputs; Python checks this |
| `gap_fill.md` (optional, live mode only) | On request | — | Uses WebSearch/WebFetch to find missing items; saves every page to `raw/web/` with its URL and date; **disabled in backtest mode and for past levels**, because it would leak post-cut-off information |

### 9.2 Launcher (`fdaengine/agent/claude_runner.py`)

A Python port of APEX's `run-review.ps1`, simplified for on-demand use:

```python
cmd = [claude_exe, "-p", prompt_text,
       "--model", "opus",
       "--output-format", "stream-json", "--verbose",   # live progress in the UI
       "--allowedTools", "Read,Glob,Grep,Write",
       "--disallowedTools", "Bash,WebFetch,WebSearch",  # gap_fill.md alone enables Web*
       "--max-turns", "80"]
subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=PIPE, stderr=STDOUT)
```

- Finds `%USERPROFILE%\.local\bin\claude.exe`, falling back to `claude` on PATH; empty stdin (APEX's stall fix).
- **Failure detection** with APEX's signatures. Each one is shown in the UI with a fix and a retry button:
  - auth failure → "run `claude login`";
  - usage limit → "retry after the reset time";
  - non-zero exit;
  - **missing/invalid output file**.
- **Schema + quote validation, then a repair loop:** on failure, one re-run with the exact errors listed (max 2).
- **Lock file** per case, so two runs can't write to the same folder.
- **Live progress:** stream-json events become UI status lines.
- **Safety:** the run reads untrusted downloaded documents, so unlike APEX it does **not** use
  `--dangerously-skip-permissions`. It has file tools only, writes are limited to `cases/<slug>/` by
  `.claude/settings.json` rules, and Python treats everything Claude writes as data to be checked.
- No Task Scheduler, health files or nightly batches (D6).

---

## 10. Online case pipeline

### 10.1 Resolve & timeline

1. Normalise names (RxNorm/UNII: brand, generic, code names); map sponsor → ticker.
2. Find the trials (ClinicalTrials.gov), applications (Drugs@FDA/openFDA), press releases (EDGAR), AdComs and CRLs.
3. **Confirmation screen:** you tick which trials and applications belong to this programme.
4. Build the **event timeline** (rule-based parsers + Claude extraction, reconciled). This gives the levels reached
   and their cut-off dates, which drive the level selector.

### 10.2 Case folder

```
cases/
  suzetrigine_dpn-pain_L2_2026-10-29/
    case.json              # inputs, level, cut-off, mode, confirmed IDs, scenario assumptions, model version
    manifest.json          # every file: url, source, sha256, retrieved_at, source_date, within_cutoff
    timeline.json          # programme events with sources
    raw/  ctgov/  edgar/  pubmed/  fda_adcom/  crl/  inspections/  orphan/  opentargets/  market/  web/
    text/                  # page-tagged text (PyMuPDF; ocrmypdf + Tesseract for scans)
    tables/                # pdfplumber/camelot CSVs
    extracted/  structured.json  fields.json  analyst_flags.json  conflicts.json
    analysis/analyst_notes.md
    features.json          # level-specific feature vector + imputation flags
    prediction.json        # posterior summaries, contributions, comparables, OOD, probability path
    payer.json             # approval-by-date curve, label probabilities, eligible population, analog prices, uptake, budget impact
    market.json            # investor module (built last): ticker, next catalyst, reaction sizes, market-implied p
    report/  report.html  report.pdf  memo.md
    logs/  claude_extract.jsonl  pipeline.log
```

Fetcher: `httpx` + `tenacity`, per-host rate limits, on-disk cache, SEC-compliant User-Agent, Playwright for
JS-rendered pages. Downloaded files are untrusted data: parsed, never executed.

### 10.3 Validate & reconcile

- Priority order: **structured values (ClinicalTrials.gov/Drugs@FDA) > rule-based parsers > Claude**. Conflicts go to
  `conflicts.json`, then to the review screen (editable; edits logged as overrides).
- Consistency checks:
  - Is the p-value consistent with the estimate and CI? (z = ln(HR)/SE, SE = (ln U − ln L)/3.92)
  - Does the CI contain the estimate? Do the arm Ns add up?
  - Do rates recomputed from counts match the stated percentages?
  - Are all sources dated ≤ the cut-off?
- Claude fields whose quote isn't found verbatim in the source text are rejected.

### 10.4 Predict, explain, flag

- **Posterior** P(approval) at the chosen level, plus P(next action approval) at L3+, with 95% credible intervals;
  the approval-date distribution; label-characteristic probabilities.
- **Confidence rating**, combining:
  - interval width;
  - completeness of high-importance fields;
  - conflict count;
  - **out-of-distribution score** (Mahalanobis / isolation forest);
  - Bayesian vs GBM disagreement.
- **Contribution waterfall:** level base rate → each feature's log-odds contribution → final.
- **Probability path** across all reached levels, annotated with the event behind each change.
- **Sensitivity tornado + scenarios** (e.g. "if the AdCom votes 4–10 against").
- **Comparable historical programmes** (Gower nearest neighbours) with outcomes, time to approval and resulting
  labels (plus stock reactions once the investor module exists).
- **Flags with evidence**, for example:
  > 🔴 **PDUFA extended 3 months after a major amendment.** Historically, programmes with an extension at L5 were
  > approved on that cycle 71% of the time (n = 96) vs 84% without. Contribution: −0.5 logit.

  These come in Critical / Warning / Info tiers, plus separate **analyst flags** (qualitative, not in the model) and
  **data flags** (imputed, conflicting, out-of-distribution, sources excluded by the cut-off).

---

## 11. Report (v1 style, upgraded)

Jinja2 + v1's CSS (`report/static/report.css`) + Plotly. Sections:

The report has three **audience profiles** (selected in the UI). Core sections are shared; the audience panel and memo
change.

1. **Header:** drug, indication, **level + cut-off date**, mode; big P(approval); 95% credible interval; confidence
   badge; probability bar; P(next action) at L3+.
2. **Probability path** chart across levels + **approval-date distribution** (cumulative "P(approved by…)" curve).
3. **Audience panel:**
   - *Payer:*
     - P(approved by plan year start);
     - expected label (boxed warning / REMS / accelerated / narrow);
     - eligible population per 100k members;
     - analog price range;
     - uptake curve;
     - budget impact (PMPM, with an interval).
   - *Investor (once the module is built):*
     - ticker and market cap;
     - next catalyst;
     - historical approval/CRL stock reactions;
     - market-implied p and edge;
     - backtested strategy stats.
   - *Neutral:* no panel; core sections only.
4. **Key positive drivers / key risk factors** (from model contributions).
5. **Risk flags:** model-backed, analyst, data quality.
6. **Contribution waterfall.**
7. **Posterior distribution** (+ the L1 stage decomposition).
8. **Sensitivity tornado + scenario table.**
9. **Comparable programmes.**
10. **Evidence table:** every field with its value, source, page and quote, linked to the file in `raw/`.
11. **"What drives approval at this level"**: the global forest plot from the model card.
12. **Memo** for the selected audience: a payer budget/formulary memo or an investment committee memo (Claude memo
    run, numbers checked by Python).
13. **Model card summary + disclaimer:**
    - model version and training N;
    - per-level Brier score/AUROC;
    - timing and label calibration;
    - payer backtest headline (and the market-backtest headline once built).

Exports: self-contained `report.html`; `report.pdf` via Playwright print-to-PDF.

---

## 12. Streamlit UI

1. **New analysis:**
   - drug, indication, optional NCT IDs;
   - **level selector:** L1 Phase 2 · L2 Phase 3 readout · L3 Filed · L4 Post-AdCom · L5 Pre-PDUFA · L6 Resubmission ·
     Auto. Levels not yet reached are greyed out;
   - mode: live / backtest (pick a date) / scenario;
   - **audience:** payer / investor (once built) / neutral; payer settings: plan-year start, member mix, optional
     customer data file;
   - gap-fill toggle (live mode only).
2. **Confirm programme:** candidate trials/applications with match scores and checkboxes; the detected timeline is
   shown so you can see which levels are available.
3. **Scenario inputs** (scenario mode only): assumed results for levels not yet reached.
4. **Live progress:** download → OCR → Claude extraction (streamed) → validation → model → market.
5. **Review evidence:** editable fields with citations and conflicts highlighted.
6. **Report:** embedded report, downloads, re-run with edits, "run all levels" for the probability path.
7. **Pipeline view (payer):** upload or pick a list of pipeline drugs → aggregate plan-year budget impact with an
   interval, top uncertainty contributors, decision-date calendar, and watch-trigger alerts. CSV/Excel export.
8. **Watchlist / case library:** saved cases sorted by next decision date; one click re-runs a case when new events
   appear.
9. **Model explorer:** model card, hypothesis table, partial-effect curves, calibration per level, timing/label
   accuracy, payer backtest (and market backtest once built).

Claude and model runs happen in background subprocesses, so the UI stays responsive.

---

## 13. Repository layout

```
fda_approval_prob_engine/
├── PLAN.md
├── CLAUDE.md                   # standing rules for headless runs (cite everything, cut-off rule, write scope, no probabilities)
├── README.md
├── pyproject.toml              # uv, Python 3.12
├── .env.example                # OPENFDA_API_KEY, AACT creds, SEC_USER_AGENT, APCA_* (market data); no Anthropic key
├── .claude/settings.json       # Read/Glob/Grep; Write only under cases/**; deny Bash/Web by default
├── legacy/webapp_v1.html
├── app/streamlit_app.py
├── prompts/                    # extract_case.md, analyst_review.md, write_memo.md, gap_fill.md, repair.md
├── schemas/                    # fields.schema.json, analyst_flags.schema.json, timeline.schema.json
├── fdaengine/
│   ├── config.py, schema.py
│   ├── resolve/                # rxnorm.py, unii.py, mesh.py, programme_matcher.py, ticker.py
│   ├── sources/                # aact.py, ctgov.py, edgar.py, drugsfda.py, openfda.py, adcom.py, crl.py,
│   │                           # inspections.py, orphan.py, pubmed.py, opentargets.py, market.py
│   ├── timeline/               # press_release_parser.py (rule-based), events.py, levels.py (cut-off logic)
│   ├── acquire/                # fetcher.py, case_folder.py, manifest.py, cutoff.py
│   ├── prepare/                # pdf_text.py, ocr.py, tables.py, ctgov_structured.py
│   ├── agent/                  # claude_runner.py, stream_parser.py, lock.py
│   ├── validate/               # quotes.py, consistency.py, reconcile.py
│   ├── features/               # build.py (per level), stats_transforms.py, assurance.py, impute.py
│   ├── dataset/                # programmes.py, labels.py, landmarks.py, censoring.py
│   ├── model/                  # univariate.py, glm.py, firth.py, gee.py, competing_risks.py, selection.py,
│   │                           # bayes_landmark.py, gbm.py, calibration.py, validate.py, registry.py
│   ├── payer/                  # analogs.py, pricing.py, uptake_bass.py, population.py, budget_impact.py, pipeline.py
│   ├── market/                 # (built last) event_study.py, edge_regression.py, strategy_backtest.py, implied_prob.py
│   ├── predict/                # predictor.py, explain.py, path.py, scenarios.py, comparables.py, ood.py, flags.py
│   └── report/                 # render.py, charts.py, numbers_check.py, templates/report.html.j2, static/report.css
├── scripts/                    # 01…13 (§5; 09 landmark, 10 timing, 11 label outcomes), run_case.py (CLI equivalent of the UI), run_pipeline.py (payer portfolio), claude_sample_extract.py (optional, manual)
├── notebooks/
├── data/  models/  cases/  exports/  logs/      (data/cases/logs git-ignored)
└── tests/                      # parsers, stat transforms, cut-off enforcement, golden-file cases
```

**Stack:**
- **Environment:** `uv`, **Python 3.12** (the machine has 3.14, but PyMC/PyTensor/LightGBM Windows wheels may lag).
- **Data:** pandas/polars, DuckDB, pyarrow.
- **Networking:** httpx, tenacity, requests-cache, Playwright.
- **Documents:** PyMuPDF, pdfplumber, camelot, ocrmypdf + Tesseract.
- **Statistics:** scipy, statsmodels (GLM, GEE, MICE), scikit-learn, firthlogist, lifelines, PyMC, ArviZ.
- **ML:** LightGBM, SHAP.
- **Market data:** alpaca-py / yfinance.
- **Output/UI:** Jinja2, Plotly, Streamlit.
- **Quality:** pytest, ruff, jsonschema, pydantic.

---

## 14. Build phases

| Phase | Deliverable | Done when |
|---|---|---|
| **P0: Foundations** | uv env, layout, schemas, cached fetcher, `CLAUDE.md`, `.claude/settings.json`, move v1 to `legacy/` | Tests pass; one ClinicalTrials.gov record lands in a case folder |
| **P1: Resolve + timeline + acquire** | Resolver, source clients, rule-based press-release parser, levels/cut-off logic, manifest | For 10 known drugs, the detected timeline matches hand-checked dates; choosing L1 vs L5 changes exactly which files are downloaded |
| **P2: Claude case extraction** | `claude_runner.py`, prompts, schema + quote validation, repair loop, reconciliation | 20 hand-checked cases: ≥ 95% accuracy on key fields; auth/usage-limit failures handled cleanly in the UI |
| **P3: Historical dataset** | Scripts 01–05: programmes, timelines, labels, censoring, landmark rows | Gold-set accuracy measured for grouping, event dates and labels |
| **P4: Statistical inference** | Scripts 06–07 (frequentist): EDA, FDR screening, GLM/Firth/GEE, competing risks, stability selection | Hypothesis table H1–H17 complete per level |
| **P5: Production model** | Bayesian landmark model + Y_NEXT model + timing model + label models + GBM challenger; validation; calibration; benchmarks | Calibration slope 0.8–1.2 at every level; beats all benchmarks on Brier score; timing calibrated at 12/24 months |
| **P6: Payer core** (re-scoped 2026-11-01, D9; P6a–P6c done, see Build status) | **P6a timing model:** P(approved by date d) per level, competing risks (approval vs failure), PDUFA-anchored from L3. **P6b new-indication labels:** supplement approvals for drugs already on the market (label accuracy 68% → ≥ 85% on the gold set), modelled as a second cohort. **P6c label breadth:** P(boxed warning), P(REMS), P(accelerated approval), P(label narrower than the trial population), conditional on approval. **P6d live cases:** context features (TA, sponsor history, moiety approved), switch the app to the fitted model, approval-date curve and label panel in the report | P(approved within 12/24/36 months) calibrated per level (temporal holdout); supplement labels ≥ 85% on the gold set; label models beat base rates; `prediction.json` carries timing and label outputs |
| **P7: Report (neutral profile)** | Template, charts, memo run + number check, PDF | Renders for 6 backtest drugs (approved; P3 fail; P3 success → CRL; CRL → approval; oncology; rare disease) at every level |
| **P8: UI** | Streamlit screens, level selector, scenarios, watchlist, model explorer | Type a drug, pick a level → report, without touching the terminal |
| **P9: Payer module** | Analog library (script 09), pricing, Bass uptake, population, budget impact, pipeline view, payer report profile + memo, plan-year backtest | Backtested plan year: 90% budget intervals cover the actual result; P(approved by date) is calibrated |
| **P10: Hardening** | Golden-file regression tests, one-command retrain, docs | A new model version can be retrained and swapped in with one command |
| **P11: Investor module + market strategy test** — **paused (D9)** | Event study, edge regression (M1), strategy backtest with deflated Sharpe, implied p (if options data is bought), investor report profile, optional APEX export | Not scheduled: only if the user asks |

**Fast first win:** P0 → P1 → P2 → P7 with a temporary model (literature phase-transition priors + assurance) → P8.
That gives an end-to-end demo with the level selector early. P3–P6 then replace the temporary model without changing
the UI.

**Order from 2026-11-01 (D9, payer first):** P6a timing → P6b new-indication labels → P6c label breadth → P6d live
cases and app switch → P9 payer module (analog pricing, uptake, eligible population, budget impact, pipeline view,
plan-year backtest) → P10 hardening. The investor module (P11) is paused.

---

## 15. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Look-ahead leakage | Per-level cut-off filter in code + tests; record-history reconstruction; gap-fill disabled for past levels/backtests; post-decision FDA documents never used |
| Event dates wrong (rule-based press-release parsing) | Gold-set accuracy; multiple sources per event; parser confidence carried into features; manual confirmation in live cases |
| Censoring bias | IPCW, competing risks, fully-resolved sensitivity analysis |
| Under-reported failures (private sponsors) | Multiple label sources + `label_confidence`; ClinicalTrials.gov termination status; public-sponsor sensitivity analysis |
| Repeated rows per programme (landmarks) | Programme random effect / clustered SEs; grouped CV splits by programme |
| Overfitting | Shrinkage (horseshoe, partial pooling across levels), stability selection, pre-registered hypotheses, temporal validation |
| Analog prices are a weak guide for first-in-class / novel-modality drugs | Price shown as a wide distribution; ICER benchmark cross-check; customer price data accepted as an input; flagged as low-confidence when there are few analogs |
| Free CMS data reflects Medicare/Medicaid, not commercial insurers | Clearly labelled as public benchmarks; customer claims/discount files plug in |
| Correlated drugs make pipeline totals look more certain than they are | Shared class-level random effects in the Monte Carlo aggregation |
| Market backtest biases | Delisted tickers, overlapping events, after-hours timing, transaction costs; deflated Sharpe for multiple strategy trials |
| "Edge" disappears when markets already price public information | M1 tested honestly with CIs; report states when no edge is found |
| Claude usage limits / auth expiry | Claude only runs for live cases; clear UI errors with fixes; nothing in training depends on Claude |
| Prompt injection from downloaded documents | No shell/web tools (except opt-in gap-fill), write scope limited to the case folder, schema + quote validation |
| Regime shifts (2025+ FDA policy changes) | Era effects, rolling backtests, warning when predicting into periods with little training data |

---

*Not investment, actuarial or medical advice. Model outputs are probabilistic estimates from public data and require
independent review before any trading, pricing or coverage decision.*
