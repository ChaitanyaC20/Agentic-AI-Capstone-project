"""
Shared utilities for the Sahayak agentic pipeline.

Contains three groups:
  1. Drug name data      — Indian brand → INN mapping for the FDA tool
  2. Symptom mapping     — narrative → canonical 11-term vocabulary for RAG
  3. Hybrid RAG engine   — embedding cache + dedup + hybrid search
"""

import os
import re
import sqlite3

# -- Keep notebook output clean: silence one-time HF/torch import advisories
# from the sentence-transformers model used by the hybrid-RAG tool.
import logging as _logging

for _k, _v in {
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_VERBOSITY": "error",
    "TRANSFORMERS_VERBOSITY": "error",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}.items():
    os.environ.setdefault(_k, _v)

for _name in (
    "huggingface_hub", "huggingface_hub.utils._http", "transformers",
    "transformers.modeling_utils", "sentence_transformers", "datasets",
    "torch", "torch.distributed.elastic.multiprocessing.redirects", "torchao",
):
    _logging.getLogger(_name).setLevel(_logging.ERROR)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Drug name data
# ─────────────────────────────────────────────────────────────────────────────

INDIA_DRUG_ALIASES: dict = {
    "crocin":     "paracetamol",   "dolo":       "paracetamol",
    "dolo 650":   "paracetamol",   "calpol":     "paracetamol",
    "panadol":    "paracetamol",   "p 500":      "paracetamol",
    "combiflam":  "ibuprofen",     "brufen":     "ibuprofen",
    "nurofen":    "ibuprofen",
    "disprin":    "aspirin",       "ecosprin":   "aspirin",
    "metformin":  "metformin",     "glycomet":   "metformin",
    "glucophage": "metformin",
    "augmentin":  "amoxicillin",   "mox":        "amoxicillin",
    "azithral":   "azithromycin",  "zithromax":  "azithromycin",
    "azee":       "azithromycin",
    # India-only — not in FDA database, handled separately
    "nimesulide": "_india_only_nimesulide",
    "nimulid":    "_india_only_nimesulide",
    "nise":       "_india_only_nimesulide",
}

INDIA_ONLY_DRUG_WARNINGS: dict = {
    "_india_only_nimesulide": {
        "note": (
            "Nimesulide is NOT FDA-approved; banned in the EU and US due to hepatotoxicity. "
            "In India: avoid in children under 12, liver disease, kidney disease, and concurrent "
            "NSAID use. Combination with alcohol can cause acute liver failure. "
            "If patient has jaundice, dark urine, or severe abdominal pain after this drug: ER."
        ),
        "risk_level": "HIGH",
        "source":     "India_CDSCO_clinical_reference",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Symptom synonym mapping (11-term canonical vocabulary)
#
# The case DB uses exactly 11 symptom terms. Patient narratives use everyday
# language. This map bridges the two so the hybrid search has sparse signal.
# ─────────────────────────────────────────────────────────────────────────────

SYMPTOM_SYNONYMS: dict = {
    "abdominal pain":      ("abdominal", "abdomen", "stomach", "belly", "tummy"),
    "blurred vision":      ("blurred", "blurry", "vision", "eyesight", "see clearly"),
    "chest pain":          ("chest", "heart racing", "heart is racing", "heart pounding",
                            "heart is pounding", "heart is beating", "racing heart", "palpitation"),
    "diarrhea":            ("diarrhea", "diarrhoea", "loose stool", "loose motion"),
    "dizziness":           ("dizzy", "dizziness", "lightheaded", "light-headed", "giddy", "faint"),
    "fever":               ("fever", "burning up", "high temperature", "chills", "shivering",
                            "febrile", "feeling hot", "really hot", "so hot",
                            "feeling cold", "really cold", "so cold", "shaking"),
    "headache":            ("headache", "head hurts", "head is pounding", "migraine", "head pain"),
    "joint pain":          ("joint", "body ache", "aching all over"),
    "rash":                ("rash", "hives", "red spots", "itchy skin", "skin eruption"),
    "shortness of breath": ("breath", "breathing", "breathe", "wheez", "gasping", "cough"),
    "vomiting":            ("vomit", "throwing up", "threw up", "nausea", "nauseous", "puking"),
}


_NEGATORS = frozenset((
    "no", "not", "without", "denies", "denied", "never", "nil",
    "doesn't", "hasn't", "isn't", "wasn't", "aren't", "don't",
))

# "can't breathe" / "not breathing" mean the symptom IS present, so the breath
# canonical is exempt from the generic negation window. Explicit absence is
# instead detected via these phrases.
_BREATH_ABSENCE_PHRASES = (
    "no shortness of breath", "no difficulty breathing", "no trouble breathing",
    "no breathing problem", "breathing normally", "breathing fine",
    "breathing is normal", "breathing is fine",
)

# Body-location words that only indicate "abdominal pain" when a pain word
# appears nearby — "rash spreading from her stomach" is a location, not pain.
_LOCATION_ONLY_WORDS = frozenset(("abdominal", "abdomen", "stomach", "belly", "tummy"))
_PAIN_CONTEXT_RE = re.compile(
    r"pain|ache|aching|hurt|cramp|burn|upset|discomfort|tender|sore"
)
_WORD_RE = re.compile(r"[a-z']+")


def _is_negated(t: str, pos: int) -> bool:
    """True if a match at character `pos` has a negator within the 4 preceding words.

    Negation scope ends at a clause boundary (, . ; : !) — "no fever, mild rash"
    negates fever but not rash. Same rule as the NegEx algorithm.
    """
    window = t[max(0, pos - 30):pos]
    clause = re.split(r"[,.;:!?]", window)[-1]
    words = _WORD_RE.findall(clause)
    return any(w in _NEGATORS for w in words[-4:])


def _has_pain_context(t: str, pos: int, pat_len: int) -> bool:
    """True if a pain/discomfort word appears within ±40 chars of the match."""
    lo, hi = max(0, pos - 40), pos + pat_len + 40
    return bool(_PAIN_CONTEXT_RE.search(t[lo:hi]))


def extract_canonical_symptoms(text: str) -> list:
    """Map a free-text narrative onto the DB's 11-term symptom vocabulary.

    Negation-aware: "no fever, no vomiting" does not fire fever or vomiting.
    A symptom fires only if at least one occurrence is non-negated.
    """
    t = text.lower()
    found = []
    for canon, pats in SYMPTOM_SYNONYMS.items():
        if canon == "shortness of breath":
            # Negated breathing ("can't breathe") = distress, so skip the
            # generic negation check; honour explicit absence phrases instead.
            t_clean = t
            for phrase in _BREATH_ABSENCE_PHRASES:
                t_clean = t_clean.replace(phrase, "")
            if any(p in t_clean for p in pats):
                found.append(canon)
            continue

        for p in pats:
            start = 0
            while (pos := t.find(p, start)) != -1:
                start = pos + 1
                if _is_negated(t, pos):
                    continue
                if p in _LOCATION_ONLY_WORDS and not _has_pain_context(t, pos, len(p)):
                    continue
                found.append(canon)
                break
            if canon in found:
                break
    return sorted(found)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hybrid RAG engine
# ─────────────────────────────────────────────────────────────────────────────
# Three problems solved here:
#   A. Query–document asymmetry: embed the canonical rewrite, not the raw narrative
#   B. Fake consensus from duplicates: deduplicate 500 rows → 288 unique groups
#   C. Unweighted majority vote: use similarity-weighted votes instead
# ─────────────────────────────────────────────────────────────────────────────

_SIM_THRESHOLD = 0.42       # below this, pure dense retrieval is too noisy

# Module-level caches — loaded once per process, not once per tool call
_ST_MODEL      = None
_EMBED_MATRIX  = None
_EMBED_IDS:   list = []
_EMBED_DB:    str  = ""
_CASE_GROUPS: list = []
_GROUP_DB:    str  = ""


def _get_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        # Contain the one-time torch/sentence-transformers import noise so it
        # never lands in a notebook cell. These are import messages, not errors.
        import contextlib
        import io as _io
        import warnings as _warnings
        try:
            with _warnings.catch_warnings(), \
                    contextlib.redirect_stderr(_io.StringIO()):
                _warnings.simplefilter("ignore")
                from sentence_transformers import SentenceTransformer
                _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            pass
    return _ST_MODEL


def _get_embed_matrix(db_path: str):
    """Load and cache the 500×384 float32 embedding matrix from SQLite."""
    global _EMBED_MATRIX, _EMBED_IDS, _EMBED_DB
    if _EMBED_MATRIX is not None and _EMBED_DB == db_path:
        return _EMBED_MATRIX, _EMBED_IDS
    try:
        import numpy as np
    except ImportError:
        return None, []
    conn = sqlite3.connect(db_path)
    try:
        has = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='case_embeddings'"
        ).fetchone()
        if not has:
            return None, []
        rows = conn.execute(
            "SELECT case_id, vector FROM case_embeddings ORDER BY case_id"
        ).fetchall()
        if not rows:
            return None, []
        ids = [r[0] for r in rows]
        dim = len(rows[0][1]) // 4
        matrix = (
            __import__("numpy").frombuffer(b"".join(r[1] for r in rows), dtype="float32")
            .reshape(len(ids), dim)
        )
        _EMBED_MATRIX, _EMBED_IDS, _EMBED_DB = matrix, ids, db_path
        return matrix, ids
    finally:
        conn.close()


def _get_case_groups(db_path: str) -> list:
    """Deduplicate 500 rows into unique symptom-combo groups (cached).

    All duplicates have the same label — verified: 0 inconsistent groups.
    Operating at the group level prevents a single case from voting k times.
    """
    global _CASE_GROUPS, _GROUP_DB
    if _CASE_GROUPS and _GROUP_DB == db_path:
        return _CASE_GROUPS
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("""
            SELECT symptoms, urgency_category, COUNT(*) n, MIN(id),
                   MAX(urgency_reasoning), MAX(red_flags)
            FROM triage_cases
            GROUP BY symptoms, urgency_category
        """).fetchall()
    finally:
        conn.close()
    groups = [
        {
            "symptoms":    sym,
            "symptom_set": frozenset(s.strip().lower() for s in sym.split(",")),
            "decision":    label,
            "n_cases":     n,
            "anchor_id":   aid,
            "reasoning":   reasoning,
            "red_flags":   red_flags,
        }
        for sym, label, n, aid, reasoning, red_flags in rows
    ]
    _CASE_GROUPS, _GROUP_DB = groups, db_path
    return groups


def hybrid_search(symptoms_text: str, top_n: int, db_path: str):
    """Hybrid sparse + dense retrieval over deduplicated case groups.

    score = 0.5 × Jaccard symptom overlap   (sparse — exact canonical terms)
          + 0.5 × cosine similarity          (dense  — semantic embedding)

    The query embedded is the REWRITTEN canonical form, not the raw narrative,
    so it is symmetric with how documents were embedded at setup time.

    Returns: (extracted_symptoms: list, top: list of (score, jac, cos, group))
    """
    groups = _get_case_groups(db_path)
    if not groups:
        return [], []

    extracted = extract_canonical_symptoms(symptoms_text)
    qset = set(extracted)

    # Dense layer
    dense = {}
    matrix, ids = _get_embed_matrix(db_path)
    model = _get_model() if matrix is not None else None
    if model is not None:
        import numpy as np  # noqa: F401 — already a hard dep of sentence-transformers
        q_text = ", ".join(extracted) if extracted else symptoms_text
        q_vec  = model.encode([q_text], normalize_embeddings=True)[0]
        sims   = matrix @ q_vec
        dense  = {ids[i]: float(sims[i]) for i in range(len(ids))}

    # Score every group
    scored = []
    for g in groups:
        jac   = (len(qset & g["symptom_set"]) / len(qset | g["symptom_set"])) if qset else 0.0
        cos   = dense.get(g["anchor_id"], 0.0)
        score = (0.5 * jac + 0.5 * cos) if qset else cos
        scored.append((score, jac, cos, g))
    scored.sort(key=lambda x: -x[0])

    if not qset:
        # Zero vocabulary overlap means the system has no grounded basis for a
        # decision — cosine similarity alone is too noisy here (the 11-term
        # vocabulary defines the knowledge boundary). Abstain.
        return extracted, []
    top = [s for s in scored[:top_n] if s[1] > 0]   # must share >=1 canonical term
    return extracted, top
