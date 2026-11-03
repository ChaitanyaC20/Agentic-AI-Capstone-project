"""Dataset utilities for the Sahayak Health AI agent capstone.

Learners should import this file instead of editing it. The helper functions
standardize column names, provide a small offline fallback dataset, and create
deterministic evaluation samples for the capstone notebooks.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

# Silence HuggingFace Hub's anonymous-access advisory: the capstone dataset is
# public, so no HF_TOKEN is needed and the warning is just noise.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DATASETS_VERBOSITY", "error")
for _name in ("huggingface_hub", "huggingface_hub.utils._http", "datasets"):
    logging.getLogger(_name).setLevel(logging.ERROR)


# WAIT = can manage at home / pharmacy / routine primary care.
# DOCTOR = needs medical attention soon, usually within 24 hours.
# ER = needs emergency care immediately.
TRIAGE_MAP = {
    "Fungal infection": "WAIT",
    "Allergy": "WAIT",
    "GERD": "WAIT",
    "Chronic cholestasis": "DOCTOR",
    "Drug Reaction": "DOCTOR",
    "Peptic ulcer diseae": "DOCTOR",
    "AIDS": "DOCTOR",
    "Diabetes": "DOCTOR",
    "Gastroenteritis": "WAIT",
    "Bronchial Asthma": "DOCTOR",
    "Hypertension": "DOCTOR",
    "Migraine": "WAIT",
    "Cervical spondylosis": "WAIT",
    "Paralysis (brain hemorrhage)": "ER",
    "Jaundice": "DOCTOR",
    "Malaria": "DOCTOR",
    "Chicken pox": "WAIT",
    "Dengue": "DOCTOR",
    "Typhoid": "DOCTOR",
    "hepatitis A": "WAIT",
    "Hepatitis B": "DOCTOR",
    "Hepatitis C": "DOCTOR",
    "Hepatitis D": "DOCTOR",
    "Hepatitis E": "DOCTOR",
    "Alcoholic hepatitis": "DOCTOR",
    "Tuberculosis": "DOCTOR",
    "Common Cold": "WAIT",
    "Pneumonia": "ER",
    "Dimorphic hemmorhoids(piles)": "WAIT",
    "Heart attack": "ER",
    "Varicose veins": "WAIT",
    "Hypothyroidism": "DOCTOR",
    "Hyperthyroidism": "DOCTOR",
    "Hypoglycemia": "ER",
    "Osteoarthristis": "WAIT",
    "Arthritis": "WAIT",
    "(vertigo) Paroymsal  Positional Vertigo": "WAIT",
    "Acne": "WAIT",
    "Urinary tract infection": "DOCTOR",
    "Psoriasis": "WAIT",
    "Impetigo": "WAIT",
}

# The 2024+ gretelai/symptom_to_diagnosis release renamed two labels and uses
# lowercase disease names. Aliases keep the provided mapping authoritative.
TRIAGE_ALIASES = {
    "gastroesophageal reflux disease": "GERD",
    "peptic ulcer disease": "Peptic ulcer diseae",  # original dataset typo
}

_TRIAGE_LOOKUP = {k.lower(): v for k, v in TRIAGE_MAP.items()}
_TRIAGE_LOOKUP.update({alias: TRIAGE_MAP[target] for alias, target in TRIAGE_ALIASES.items()})


def map_diagnosis_to_triage(diagnosis: str) -> str | None:
    """Case-insensitive diagnosis -> WAIT/DOCTOR/ER lookup. None if unknown."""

    return _TRIAGE_LOOKUP.get(str(diagnosis).strip().lower())


@dataclass(frozen=True)
class OfflineCase:
    symptom_text: str
    diagnosis: str


OFFLINE_CASES = [
    OfflineCase("itching skin rash nodal skin eruptions dischromic patches", "Fungal infection"),
    OfflineCase("continuous sneezing shivering chills watering from eyes", "Allergy"),
    OfflineCase("stomach pain acidity ulcers on tongue vomiting cough chest pain", "GERD"),
    OfflineCase("vomiting sunken eyes dehydration diarrhoea", "Gastroenteritis"),
    OfflineCase("headache blurred and distorted vision excessive hunger stiff neck", "Migraine"),
    OfflineCase("skin rash pus filled pimples blackheads scurring", "Acne"),
    OfflineCase("burning micturition bladder discomfort foul smell of urine", "Urinary tract infection"),
    OfflineCase("high fever sweating headache nausea diarrhoea muscle pain", "Malaria"),
    OfflineCase("yellowish skin dark urine nausea loss of appetite abdominal pain", "Jaundice"),
    OfflineCase("weight loss restlessness lethargy irregular sugar level blurred vision", "Diabetes"),
    OfflineCase("cough high fever breathlessness sweating malaise phlegm chest pain", "Pneumonia"),
    OfflineCase("chest pain breathlessness sweating vomiting", "Heart attack"),
    OfflineCase("weakness of one body side altered sensorium headache vomiting", "Paralysis (brain hemorrhage)"),
    OfflineCase("fatigue anxiety sweating headache nausea blurred vision", "Hypoglycemia"),
    OfflineCase("cough high fever breathlessness family history mucoid sputum", "Tuberculosis"),
    OfflineCase("continuous sneezing chills fatigue cough high fever headache", "Common Cold"),
    OfflineCase("joint pain vomiting fatigue high fever skin rash", "Dengue"),
    OfflineCase("chills vomiting fatigue high fever headache belly pain", "Typhoid"),
    OfflineCase("mild fever yellowing of eyes muscle pain loss of appetite", "hepatitis A"),
    OfflineCase("stomach pain ulcers on tongue vomiting indigestion", "Peptic ulcer diseae"),
    OfflineCase("puffy face enlarged thyroid brittle nails swollen extremeties", "Hypothyroidism"),
    OfflineCase("mood swings weight loss restlessness sweating diarrhoea", "Hyperthyroidism"),
    OfflineCase("knee pain hip joint pain swelling joints painful walking", "Osteoarthristis"),
    OfflineCase("skin rash high fever blister red sore around nose yellow crust ooze", "Impetigo"),
]


def _load_huggingface_dataset(split: str = "train") -> pd.DataFrame:
    """Load the public Hugging Face dataset when the environment supports it.

    Current gretelai/symptom_to_diagnosis schema (853 train / 212 test rows):
    - input_text: first-person natural-language symptom description
    - output_text: lowercase diagnosis label
    """

    import warnings

    from datasets import load_dataset

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = load_dataset("gretelai/symptom_to_diagnosis", split=split)
    df = pd.DataFrame(ds)

    if {"input_text", "output_text"}.issubset(df.columns):
        df = df.rename(columns={"input_text": "symptoms", "output_text": "diagnosis"})
        df = df[["symptoms", "diagnosis"]].copy()
    elif {"symptoms", "diagnosis"}.issubset(df.columns):
        df = df[["symptoms", "diagnosis"]].copy()
    else:
        raise ValueError(
            f"Unrecognized dataset columns {list(df.columns)}; expected "
            "input_text/output_text or symptoms/diagnosis."
        )

    # Validate that the diagnosis column maps to known triage labels
    # (case-insensitive, with aliases for renamed diseases).
    matched = df["diagnosis"].map(map_diagnosis_to_triage).notna().mean()
    if matched < 0.9:
        raise ValueError(
            f"Only {matched:.0%} of diagnosis labels map to the provided "
            "TRIAGE_MAP. The dataset format may have changed again. "
            "Falling back to offline dataset."
        )

    return df


def _load_offline_dataset() -> pd.DataFrame:
    """Return a small built-in dataset so notebooks still open without network."""

    return pd.DataFrame([case.__dict__ for case in OFFLINE_CASES]).rename(
        columns={"symptom_text": "symptoms"}
    )


def standardize_sahayak_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with stable capstone columns.

    Output columns:
    - symptom_text: patient symptom description
    - diagnosis: source diagnosis label
    - triage_level: WAIT | DOCTOR | ER
    """

    out = df.copy()
    if "symptom_text" not in out.columns and "symptoms" in out.columns:
        out = out.rename(columns={"symptoms": "symptom_text"})
    if "symptom_text" not in out.columns:
        raise ValueError("Expected a symptom_text or symptoms column.")
    if "diagnosis" not in out.columns:
        raise ValueError("Expected a diagnosis column.")

    out["symptom_text"] = out["symptom_text"].astype(str)
    out["diagnosis"] = out["diagnosis"].astype(str)
    out["triage_level"] = out["diagnosis"].map(map_diagnosis_to_triage)
    unmapped = out["triage_level"].isna()
    if unmapped.any():
        # Conservative default, but never silent: name the labels affected.
        print(
            f"[data_loader] {int(unmapped.sum())} rows with unmapped diagnosis "
            f"labels defaulted to DOCTOR: {sorted(out.loc[unmapped, 'diagnosis'].unique())}"
        )
        out.loc[unmapped, "triage_level"] = "DOCTOR"
    return out[["symptom_text", "diagnosis", "triage_level"]].reset_index(drop=True)


def load_sahayak_dataset(prefer_online: bool = True, split: str = "train") -> pd.DataFrame:
    """Load the Sahayak dataset with stable columns.

    The train split (853 cases) is used for notebooks and policy evaluation.
    The test split (212 cases) is reserved for the live-agent evaluation in
    eval_agent.py so prompt tuning never sees it.

    If Hugging Face or network access is unavailable, this falls back to a small
    built-in sample. The fallback is only for offline smoke tests — it is NOT
    valid for reporting capstone metrics, and a warning is printed when used.
    """

    if prefer_online:
        try:
            return standardize_sahayak_dataframe(_load_huggingface_dataset(split=split))
        except Exception as exc:
            print(
                "[data_loader] WARNING: Hugging Face load failed — using the "
                f"24-case offline fallback. Metrics on this sample are NOT "
                f"submission-grade. Reason: {exc}"
            )
    return standardize_sahayak_dataframe(_load_offline_dataset())


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    seed: int = 42,
    levels: Iterable[str] = ("WAIT", "DOCTOR", "ER"),
) -> pd.DataFrame:
    """Sample rows while keeping all triage levels represented when possible."""

    pieces = []
    sampled_indices = []
    per_level = max(1, n // len(tuple(levels)))
    for level in levels:
        subset = df[df["triage_level"] == level]
        if len(subset) == 0:
            continue
        part = subset.sample(min(per_level, len(subset)), random_state=seed)
        pieces.append(part)
        sampled_indices.extend(part.index.tolist())

    sample = pd.concat(pieces, ignore_index=True)
    remaining = max(0, min(n, len(df)) - len(sample))
    if remaining:
        pool = df.drop(index=sampled_indices, errors="ignore")
        if len(pool):
            sample = pd.concat(
                [sample, pool.sample(min(remaining, len(pool)), random_state=seed + 1)],
                ignore_index=True,
            )

    return sample.sample(frac=1, random_state=seed).reset_index(drop=True)


def get_sample_cases(n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Return a small stratified sample for Week 2/3 experiments."""

    return stratified_sample(load_sahayak_dataset(), n=n, seed=seed)


def get_test_cases(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Return a deterministic evaluation set for Week 3/4."""

    return stratified_sample(load_sahayak_dataset(), n=n, seed=seed)


def build_evaluation_dataset(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Alias for the held-out test split."""

    return get_test_cases(n=n, seed=seed)


def get_triage_map_table() -> pd.DataFrame:
    """Return the provided diagnosis -> triage mapping for comparison.

    Week 2 asks learners to propose their own mapping first. This helper is then
    used as the reference answer, similar to showing a rubric after an attempt.
    """

    return pd.DataFrame(
        [{"diagnosis": diagnosis, "triage_level": triage} for diagnosis, triage in TRIAGE_MAP.items()]
    ).sort_values("diagnosis").reset_index(drop=True)


def show_dataset_stats(df: pd.DataFrame | None = None) -> None:
    """Print a beginner-friendly dataset summary."""

    if df is None:
        df = load_sahayak_dataset()

    print(f"Total cases: {len(df)}")
    print()
    print("Triage level distribution:")
    print(df["triage_level"].value_counts().to_string())
    print()
    print("Diagnosis breakdown:")
    print(df.groupby(["triage_level", "diagnosis"]).size().to_string())


def get_emergency_cases() -> pd.DataFrame:
    """Return only ER-level cases for stress-testing the agent."""

    df = load_sahayak_dataset()
    return df[df["triage_level"] == "ER"].reset_index(drop=True)


def compute_basic_metrics(results: pd.DataFrame) -> dict:
    """Compute accuracy and per-class recall without sklearn."""

    required = {"true_triage", "predicted_triage"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Missing columns for metric computation: {sorted(missing)}")

    accuracy = float((results["true_triage"] == results["predicted_triage"]).mean())
    per_class = {}
    for level in ["WAIT", "DOCTOR", "ER"]:
        subset = results[results["true_triage"] == level]
        if len(subset) == 0:
            per_class[level] = None
        else:
            per_class[level] = float((subset["predicted_triage"] == level).mean())
    return {"accuracy": accuracy, "recall_by_triage": per_class}
