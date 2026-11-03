"""
One-time setup: parse medical_triage_500.jsonl and load into SQLite with
semantic embeddings for RAG retrieval.

Run once:
    python setup_db.py

Creates cases.db with two tables:
  triage_cases     — 500 clinical triage cases (symptoms, labels, reasoning)
  case_embeddings  — float32 L2-normalised vectors (semantic similarity search)

Dataset schema (nested structs, parsed manually):
  case_id, patient.{age,gender},
  presentation.{symptoms:list, symptom_count, duration, onset, context},
  risk_assessment.{risk_level, red_flags:list},
  triage_classification.{urgency_category, urgency_reasoning},
  safety_notice

urgency_category → Sahayak label:
  immediate → ER   |   urgent → DOCTOR   |   routine → WAIT
"""

import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

DB_PATH    = "cases.db"
JSONL_URL  = "https://huggingface.co/datasets/syntech-ai/medical-triage-500/resolve/main/medical_triage_500.jsonl"
JSONL_PATH = "medical_triage_500.jsonl"
EMBED_MODEL = "all-MiniLM-L6-v2"

URGENCY_MAP = {
    "immediate": "ER",
    "urgent":    "DOCTOR",
    "routine":   "WAIT",
}


def _download_jsonl() -> list[dict]:
    if not Path(JSONL_PATH).exists():
        print(f"Downloading dataset to {JSONL_PATH}...")
        urllib.request.urlretrieve(JSONL_URL, JSONL_PATH)
    else:
        print(f"Using cached {JSONL_PATH}")

    rows = []
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "case_id" not in obj:      # skip metadata header rows
                continue
            rows.append(obj)
    print(f"  {len(rows)} data rows loaded.")
    return rows


def _flatten(obj: dict) -> dict:
    """Convert nested struct to flat fields for SQLite."""
    patient  = obj.get("patient", {})
    pres     = obj.get("presentation", {})
    risk     = obj.get("risk_assessment", {})
    triage   = obj.get("triage_classification", {})

    symptoms_list  = pres.get("symptoms", [])
    red_flags_list = risk.get("red_flags", [])

    raw_urgency = triage.get("urgency_category", "").lower().strip()
    urgency_label = URGENCY_MAP.get(raw_urgency, raw_urgency)

    return {
        "case_id":          obj.get("case_id", ""),
        "age":              patient.get("age"),
        "gender":           patient.get("gender", ""),
        "symptoms":         ", ".join(symptoms_list),     # joined string for LIKE + embedding
        "symptom_count":    pres.get("symptom_count"),
        "duration":         pres.get("duration", ""),
        "onset":            pres.get("onset", ""),
        "context":          pres.get("context", ""),
        "risk_level":       risk.get("risk_level", ""),
        "red_flags":        ", ".join(red_flags_list),
        "urgency_category": urgency_label,                # ER / DOCTOR / WAIT
        "urgency_reasoning": triage.get("urgency_reasoning", ""),
        "safety_notice":    obj.get("safety_notice", ""),
    }


def setup_cases_db(db_path: str = DB_PATH) -> int:
    raw_rows = _download_jsonl()
    flat     = [_flatten(r) for r in raw_rows]

    conn = sqlite3.connect(db_path)

    # ── Main cases table ──────────────────────────────────────────────────────
    conn.execute("DROP TABLE IF EXISTS triage_cases")
    conn.execute("""
        CREATE TABLE triage_cases (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id           TEXT,
            age               INTEGER,
            gender            TEXT,
            symptoms          TEXT,
            symptom_count     INTEGER,
            duration          TEXT,
            onset             TEXT,
            context           TEXT,
            risk_level        TEXT,
            red_flags         TEXT,
            urgency_category  TEXT,
            urgency_reasoning TEXT,
            safety_notice     TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO triage_cases VALUES "
        "(NULL,:case_id,:age,:gender,:symptoms,:symptom_count,:duration,"
        ":onset,:context,:risk_level,:red_flags,:urgency_category,"
        ":urgency_reasoning,:safety_notice)",
        flat,
    )
    conn.commit()
    print(f"  triage_cases: {len(flat)} rows written.")

    # ── Embeddings table ──────────────────────────────────────────────────────
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("WARNING: sentence-transformers not found — skipping embeddings.")
        print("  pip install sentence-transformers  to enable semantic RAG.")
        conn.close()
        return len(flat)

    print(f"Building embeddings with {EMBED_MODEL} ...")
    symptoms_texts = [r["symptoms"] for r in flat]
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        symptoms_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalised → cosine = dot product
    )

    conn.execute("DROP TABLE IF EXISTS case_embeddings")
    conn.execute("""
        CREATE TABLE case_embeddings (
            case_id  INTEGER PRIMARY KEY,
            vector   BLOB NOT NULL
        )
    """)
    conn.executemany(
        "INSERT INTO case_embeddings VALUES (?,?)",
        [(i + 1, embeddings[i].astype("float32").tobytes()) for i in range(len(embeddings))],
    )
    conn.commit()
    conn.close()

    dim = embeddings.shape[1]
    size_kb = Path(db_path).stat().st_size // 1024
    print(f"  case_embeddings: {len(flat)} vectors ({dim}-dim float32) written.")
    print(f"\nDatabase ready: {db_path}  (~{size_kb} KB)")
    return len(flat)


if __name__ == "__main__":
    n = setup_cases_db()
    print(f"\nDone. {n} cases. Semantic RAG active.")
