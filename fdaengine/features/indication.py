"""Therapeutic area from ClinicalTrials.gov condition MeSH terms (rule-based, shared by training and live cases).

ClinicalTrials.gov maps each trial's conditions to MeSH descriptors and lists their ancestors
(`derivedSection.conditionBrowseModule`). The TA is the first matching rule below, checked against the leaf terms
and their ancestors, with a keyword fallback on the condition text for trials without MeSH terms. Area names match
`model.provisional.TA_OFFSET`.
"""

from __future__ import annotations

import re

# (TA, MeSH terms that place a trial in it). Order is precedence: a lung-cancer trial is oncology, not respiratory;
# Alzheimer's (both a nervous-system and a mental disorder in MeSH) is neurology; type 1 diabetes is metabolic.
TA_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("oncology", ("Neoplasms",)),
    ("infectious_disease", ("Infections", "Bacterial Infections and Mycoses", "Virus Diseases", "Parasitic Diseases",
                            "Communicable Diseases", "Bacterial Infections", "Mycoses")),
    ("pain", ("Pain", "Analgesia", "Headache Disorders", "Headache", "Migraine Disorders")),
    ("neurology", ("Nervous System Diseases",)),
    ("psychiatry", ("Mental Disorders", "Behavior and Behavior Mechanisms", "Substance-Related Disorders")),
    ("ophthalmology", ("Eye Diseases",)),
    ("cardiovascular", ("Cardiovascular Diseases",)),
    ("hematology", ("Hemic and Lymphatic Diseases",)),
    ("respiratory", ("Respiratory Tract Diseases",)),
    ("metabolic_endocrine", ("Nutritional and Metabolic Diseases", "Endocrine System Diseases")),
    ("immunology", ("Autoimmune Diseases", "Immune System Diseases", "Rheumatic Diseases")),
    ("gastroenterology", ("Digestive System Diseases",)),
    ("dermatology", ("Skin Diseases", "Skin and Connective Tissue Diseases")),
    ("musculoskeletal", ("Musculoskeletal Diseases",)),
    ("nephrology", ("Kidney Diseases", "Urologic Diseases", "Male Urogenital Diseases")),
    ("womens_health", ("Female Urogenital Diseases and Pregnancy Complications", "Pregnancy Complications",
                       "Female Urogenital Diseases")),
]
_TEXT_RULES: list[tuple[str, str]] = [
    ("oncology", (r"cancer|carcinoma|tumou?r|neoplasm|lymphoma|leukemi|leukaemi|myeloma|melanoma|sarcoma|glioma|"
                 r"blastoma|metasta|malignan|\b(n?sclc|crc|hcc|aml|cml|cll|mds|rcc|gist)\b")),
    ("infectious_disease", (r"infection|hiv|hepatitis|influenza|covid|sars|vaccin|bacter|viral|virus|fung|sepsis|"
                           r"pneumonia|malaria|tubercul|immuni[sz]|otitis|\blice\b|pediculosis|\brsv\b")),
    ("pain", r"\bpain|analges|migraine|headache"),
    ("psychiatry", r"depress|schizo|bipolar|anxiety|adhd|autism|psych|insomnia|addict|alcohol|opioid use|smoking"),
    ("neurology", r"alzheimer|parkinson|epilep|seizure|sclerosis|neuropath|dementia|stroke|huntington|ataxia"),
    ("ophthalmology", r"macular|retin|glaucoma|ocular|\beye|uveitis|conjunctiv|keratitis|myopia"),
    ("cardiovascular", r"heart|cardi|hypertens|atrial|coronary|thromb|arter|vascular|hyperlipid|cholesterol"),
    ("hematology", r"anemi|anaemi|hemophil|haemophil|sickle|thalass|neutropeni|thrombocytopeni|bleeding"),
    ("respiratory", r"asthma|copd|pulmonary|respirat|cystic fibrosis|bronch|lung"),
    ("immunology", r"arthritis|lupus|psoria|crohn|colitis|sj[oö]gren|spondyl|vasculitis|transplant|atopic"),
    ("metabolic_endocrine", (r"diabet|obes|metabolic|thyroid|growth hormone|lipodystroph|gout|nash|mash|"
                            r"steatohepatitis|hypoglyc|cushing|acromegal")),
    ("gastroenterology", (r"bowel|gastr|hepat|liver|cirrhosis|constipation|reflux|pancrea|celiac|esophag|vomit|"
                          r"nausea|diarrh|dyspeps|colonoscop")),
    ("dermatology", (r"dermat|eczema|acne|skin|urticaria|alopecia|vitiligo|hidradenitis|rosacea|glabellar|"
                     r"canthal|wrinkle|\bburns?\b|prurigo|prurit")),
    ("musculoskeletal", r"osteopor|osteoarth|muscul|bone|fracture|dystroph|sarcopen"),
    ("nephrology", r"kidney|renal|nephr|bladder|urinar|prostat|dialysis|hyperkal|erectile"),
    ("womens_health", (r"pregnan|contracep|endometrio|menopaus|uterine|ovar|vagin|labor|preterm|fertility|fibroid|"
                       r"hot flash|dysmenorr")),
]


def therapeutic_area(leaf_terms: list[str], ancestor_terms: list[str], texts: list[str],
                     intervention_texts: list[str] = ()) -> str:
    """`intervention_texts` (drug names) only breaks ties for trials in healthy volunteers, e.g. vaccines."""
    terms = set(leaf_terms) | set(ancestor_terms)
    for ta, keys in TA_RULES:
        if terms & set(keys):
            return ta
    blob = " ".join(texts).lower()
    for ta, pattern in _TEXT_RULES:
        if re.search(pattern, blob):
            return ta
    if re.search(r"vaccin|toxoid", " ".join(intervention_texts).lower()):
        return "infectious_disease"
    return "other"
