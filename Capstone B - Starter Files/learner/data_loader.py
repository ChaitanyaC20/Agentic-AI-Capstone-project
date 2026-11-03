"""Sahayak Health AI -- dataset utilities.

Learners should import from this file, not edit it.

Public API (all you need):
    TRIAGE_MAP                       dict[str, str]  -- 41 known diagnoses (dataset uses 22) -> WAIT/DOCTOR/ER
    load_sahayak_dataset()           -> pd.DataFrame  -- 853 train cases
    build_evaluation_dataset(n, seed)-> pd.DataFrame  -- locked 50-case eval split
"""

from __future__ import annotations
import os
import logging
import pandas as pd

# -- Silence HuggingFace Hub's "unauthenticated / anonymous" advisory ----------
# load_dataset() prints a warning when no HF_TOKEN is set. The capstone dataset
# is public, so anonymous access is fine and the warning is just noise. Suppress
# it centrally here.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DATASETS_VERBOSITY", "error")
for _name in ("huggingface_hub", "huggingface_hub.utils._http", "datasets"):
    logging.getLogger(_name).setLevel(logging.ERROR)

# -- Triage map ----------------------------------------------------------------
# WAIT   = manageable at home / pharmacy / routine primary care
# DOCTOR = needs medical attention, usually within 24 hours
# ER     = needs emergency care immediately

TRIAGE_MAP: dict[str, str] = {
    "Fungal infection":                      "WAIT",
    "Allergy":                               "WAIT",
    "GERD":                                  "WAIT",
    "Chronic cholestasis":                   "DOCTOR",
    "Drug Reaction":                         "DOCTOR",
    "Peptic ulcer diseae":                   "DOCTOR",
    "AIDS":                                  "DOCTOR",
    "Diabetes":                              "DOCTOR",
    "Gastroenteritis":                       "WAIT",
    "Bronchial Asthma":                      "DOCTOR",
    "Hypertension":                          "DOCTOR",
    "Migraine":                              "WAIT",
    "Cervical spondylosis":                  "WAIT",
    "Paralysis (brain hemorrhage)":          "ER",
    "Jaundice":                              "DOCTOR",
    "Malaria":                               "DOCTOR",
    "Chicken pox":                           "WAIT",
    "Dengue":                                "DOCTOR",
    "Typhoid":                               "DOCTOR",
    "hepatitis A":                           "WAIT",
    "Hepatitis B":                           "DOCTOR",
    "Hepatitis C":                           "DOCTOR",
    "Hepatitis D":                           "DOCTOR",
    "Hepatitis E":                           "DOCTOR",
    "Alcoholic hepatitis":                   "DOCTOR",
    "Tuberculosis":                          "DOCTOR",
    "Common Cold":                           "WAIT",
    "Pneumonia":                             "ER",
    "Dimorphic hemmorhoids(piles)":          "WAIT",
    "Heart attack":                          "ER",
    "Varicose veins":                        "WAIT",
    "Hypothyroidism":                        "DOCTOR",
    "Hyperthyroidism":                       "DOCTOR",
    "Hypoglycemia":                          "ER",
    "Osteoarthristis":                       "WAIT",
    "Arthritis":                             "WAIT",
    "(vertigo) Paroymsal  Positional Vertigo": "WAIT",
    "Acne":                                  "WAIT",
    "Urinary tract infection":               "DOCTOR",
    "Psoriasis":                             "WAIT",
    "Impetigo":                              "WAIT",
}

# The 2024+ gretelai/symptom_to_diagnosis release renamed a few labels.
# These aliases keep the TRIAGE_MAP above authoritative.
_ALIASES: dict[str, str] = {
    "gastroesophageal reflux disease": "GERD",
    "peptic ulcer disease":            "Peptic ulcer diseae",  # original typo kept
}

_LOOKUP = {k.lower(): v for k, v in TRIAGE_MAP.items()}
_LOOKUP.update({alias: TRIAGE_MAP[target] for alias, target in _ALIASES.items()})


def _map_triage(diagnosis: str) -> str | None:
    return _LOOKUP.get(str(diagnosis).strip().lower())


# -- Offline fallback (used only when HuggingFace is unavailable) --------------
_OFFLINE = [
    ("itching skin rash nodal skin eruptions dischromic patches",       "Fungal infection"),
    ("continuous sneezing shivering chills watering from eyes",         "Allergy"),
    ("stomach pain acidity ulcers on tongue vomiting cough chest pain", "GERD"),
    ("vomiting sunken eyes dehydration diarrhoea",                      "Gastroenteritis"),
    ("headache blurred vision excessive hunger stiff neck",             "Migraine"),
    ("skin rash pus filled pimples blackheads scurring",                "Acne"),
    ("burning micturition bladder discomfort foul smell of urine",      "Urinary tract infection"),
    ("high fever sweating headache nausea diarrhoea muscle pain",       "Malaria"),
    ("yellowish skin dark urine nausea loss of appetite abdominal pain","Jaundice"),
    ("weight loss restlessness irregular sugar level blurred vision",   "Diabetes"),
    ("cough high fever breathlessness sweating phlegm chest pain",      "Pneumonia"),
    ("chest pain breathlessness sweating vomiting",                     "Heart attack"),
    ("weakness of one body side altered sensorium headache vomiting",   "Paralysis (brain hemorrhage)"),
    ("fatigue anxiety sweating headache nausea blurred vision",         "Hypoglycemia"),
    ("cough high fever breathlessness family history mucoid sputum",    "Tuberculosis"),
    ("continuous sneezing chills fatigue cough high fever headache",    "Common Cold"),
    ("joint pain vomiting fatigue high fever skin rash",                "Dengue"),
    ("chills vomiting fatigue high fever headache belly pain",          "Typhoid"),
    ("mild fever yellowing of eyes muscle pain loss of appetite",       "hepatitis A"),
    ("stomach pain ulcers on tongue vomiting indigestion",              "Peptic ulcer diseae"),
    ("puffy face enlarged thyroid brittle nails swollen extremeties",   "Hypothyroidism"),
    ("mood swings weight loss restlessness sweating diarrhoea",         "Hyperthyroidism"),
    ("knee pain hip joint pain swelling joints painful walking",        "Osteoarthristis"),
    ("skin rash high fever blister red sore around nose yellow crust",  "Impetigo"),
]


# -- Public API ----------------------------------------------------------------

def load_sahayak_dataset() -> pd.DataFrame:
    """Load the full dataset (853 train cases).

    Returns a DataFrame with columns:
        symptom_text  -- first-person symptom description
        diagnosis     -- source diagnosis label
        triage_level  -- WAIT | DOCTOR | ER

    Tries gretelai/symptom_to_diagnosis from HuggingFace first.
    Falls back to a small built-in sample if offline.
    """
    try:
        import warnings
        from datasets import load_dataset
        # Anonymous access to a public dataset -- keep the console clean.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = load_dataset("gretelai/symptom_to_diagnosis", split="train")
        df = pd.DataFrame(ds)
        # HF schema uses input_text/output_text; rename to our standard
        if "input_text" in df.columns:
            df = df.rename(columns={"input_text": "symptom_text", "output_text": "diagnosis"})
        df = df[["symptom_text", "diagnosis"]].copy()
    except Exception as exc:
        print(f"[data_loader] HuggingFace unavailable -- using offline sample. ({exc})")
        df = pd.DataFrame(_OFFLINE, columns=["symptom_text", "diagnosis"])

    df["symptom_text"] = df["symptom_text"].astype(str)
    df["diagnosis"] = df["diagnosis"].astype(str)
    df["triage_level"] = df["diagnosis"].map(_map_triage)
    df.loc[df["triage_level"].isna(), "triage_level"] = "DOCTOR"  # conservative default
    return df[["symptom_text", "diagnosis", "triage_level"]].reset_index(drop=True)


def build_evaluation_dataset(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Return the locked 50-case evaluation dataset used across all.

    Draws a stratified sample from the full 853-case HuggingFace dataset,
    keeping all three triage levels (WAIT, DOCTOR, ER) represented.

    The default parameters (n=50, seed=42) are FROZEN for this capstone.
    Every notebook, every metric is measured against this
    identical 50-case set so numbers are directly comparable.

    DO NOT change n or seed when reporting capstone metrics.
    Re-sampling to get a better score is academic dishonesty.

    Parameters
    ----------
    n : int
        Number of cases to return. Default 50 -- keep this fixed so every run
        evaluates on the same sample.
    seed : int
        Random seed. Default 42 -- keep this fixed so the sample is identical
        every time you run it.

    Returns
    -------
    pd.DataFrame with columns:
        symptom_text (str) -- first-person natural language symptom description
        diagnosis    (str) -- source diagnosis label from the HuggingFace dataset
        triage_level (str) -- WAIT | DOCTOR | ER  (mapped via TRIAGE_MAP)
    """
    df = load_sahayak_dataset()
    rng = seed

    # Sample ~equal slices from each triage level, then shuffle
    per_level = max(1, n // 3)
    parts = []
    used = []
    for level in ("WAIT", "DOCTOR", "ER"):
        subset = df[df["triage_level"] == level]
        take = min(per_level, len(subset))
        part = subset.sample(take, random_state=rng)
        parts.append(part)
        used.extend(part.index.tolist())

    sample = pd.concat(parts, ignore_index=True)

    # Top-up to exactly n if needed
    remaining = n - len(sample)
    if remaining > 0:
        pool = df.drop(index=used, errors="ignore")
        if len(pool):
            extra = pool.sample(min(remaining, len(pool)), random_state=rng + 1)
            sample = pd.concat([sample, extra], ignore_index=True)

    return sample.sample(frac=1, random_state=rng).reset_index(drop=True)
