"""
Sahayak FunctionTools — the 4 tools that make the triage_decider an AGENT.

Each tool does something a plain LLM call cannot:
  1. search_symptom_cases_db  — retrieves grounded evidence from a real SQLite DB
  2. lookup_drug_safety       — calls the live FDA drug label API (no training data)
  3. parse_vitals_from_text   — deterministic regex extraction (no hallucination)
  4. calculate_india_news2    — deterministic arithmetic (wrong score = wrong triage)

ADK reads the Python type hints + docstrings to auto-generate the JSON schema
that tells the LLM when and how to call each tool.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import requests

from utils import (
    INDIA_DRUG_ALIASES,
    INDIA_ONLY_DRUG_WARNINGS,
    hybrid_search,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: Clinical case database (hybrid RAG)
# ─────────────────────────────────────────────────────────────────────────────

def search_symptom_cases_db(
    symptoms_text: str,
    top_n: int = 5,
    db_path: str = "cases.db",
) -> dict:
    """
    Query 500 past clinical cases for similar triage decisions.

    Maps the patient narrative onto canonical symptom terms, retrieves the most
    similar unique case groups using hybrid sparse+dense search, and returns a
    similarity-weighted consensus verdict.

    Returns no_match (and an honest message) if the symptoms are too dissimilar
    to anything in the database — the agent then relies on NEWS2 and judgment.

    Args:
        symptoms_text: Patient's symptom description in plain text.
        top_n: Number of unique case groups to retrieve (max 5).
        db_path: Path to the SQLite case database (run setup_db.py once).
    """
    top_n = min(top_n, 5)
    db = Path(db_path)

    if not db.exists():
        return {
            "status":  "db_not_found",
            "message": "Run 'python setup_db.py' once to build the case database.",
            "similar_cases": [],
        }

    extracted, top = hybrid_search(symptoms_text, top_n, str(db))

    if not top:
        return {
            "status":        "no_match",
            "extracted_symptoms": extracted,
            "similar_cases": [],
            "message":       "No similar past cases found. Proceed with NEWS2 and clinical judgment.",
        }

    # Build output
    cases = [
        {
            "symptoms":            g["symptoms"],
            "decision":            g["decision"],
            "n_matching_cases":    g["n_cases"],
            "match_score":         round(score, 3),
            "symptom_overlap":     round(jac, 3),
            "semantic_similarity": round(cos, 3),
            "reasoning":           g["reasoning"],
            "red_flags":           g["red_flags"],
        }
        for score, jac, cos, g in top
    ]

    # Similarity-weighted vote
    votes: dict = {}
    for score, _, _, g in top:
        votes[g["decision"]] = votes.get(g["decision"], 0.0) + score
    provisional = max(votes, key=votes.get)

    # Safety doctrine: if a higher severity level has ≥60% of the winner's
    # weight, escalate — "when in doubt, triage up"
    severity = {"WAIT": 0, "DOCTOR": 1, "ER": 2}
    consensus = provisional
    for level, weight in votes.items():
        if (severity.get(level, 0) > severity.get(consensus, 0)
                and weight >= 0.6 * votes[provisional]):
            consensus = level

    confidence = round(votes[consensus] / sum(votes.values()), 2)

    return {
        "status":               "found",
        "retrieval_method":     "hybrid_rag" if extracted else "semantic_rag",
        "extracted_symptoms":   extracted,
        "n_case_groups":        len(cases),
        "n_underlying_cases":   int(sum(g["n_cases"] for *_, g in top)),
        "consensus_decision":   consensus,
        "consensus_confidence": confidence,
        "safety_escalated":     consensus != provisional,
        "vote_breakdown":       {k: round(v, 3) for k, v in votes.items()},
        "similar_cases":        cases,
        "evidence_source":      "syntech_medical_triage_500",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: Live FDA drug safety API
# ─────────────────────────────────────────────────────────────────────────────

def lookup_drug_safety(medication_name: str) -> dict:
    """
    Look up FDA drug safety warnings for a medication.

    Handles Indian brand names (Crocin, Dolo, Combiflam, Nimesulide …) by
    mapping them to international non-proprietary names before querying.
    India-only drugs not in the FDA database use local clinical reference data.

    Call this whenever the patient mentions a drug by name — overdose or
    adverse reactions can change the triage level.

    Args:
        medication_name: Drug name, brand or generic (e.g. 'crocin', 'paracetamol').
    """
    clean  = medication_name.lower().strip()
    mapped = INDIA_DRUG_ALIASES.get(clean)

    if mapped and mapped.startswith("_india_only"):
        info = INDIA_ONLY_DRUG_WARNINGS.get(mapped, {})
        return {
            "status":     "india_specific_drug",
            "drug_queried": medication_name,
            "warnings":   info.get("note", "No data."),
            "risk_level": info.get("risk_level", "UNKNOWN"),
            "source":     info.get("source", "India_clinical_reference"),
            "note":       "Not FDA-approved. India-specific reference used.",
        }

    search_term = mapped if mapped else clean
    url = "https://api.fda.gov/drug/label.json"

    data = None
    for query in (
        f"openfda.generic_name:{search_term}",
        f"active_ingredient:{search_term}",
        search_term,
    ):
        try:
            resp = requests.get(url, params={"search": query, "limit": 1}, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("results"):
                    break
        except requests.Timeout:
            return {"status": "timeout", "drug_queried": medication_name,
                    "message": "FDA API timed out. Treat as unknown — consult a pharmacist."}
        except Exception as exc:
            return {"status": "error", "drug_queried": medication_name, "message": str(exc)}

    if not data or not data.get("results"):
        return {"status": "not_found", "drug_queried": medication_name,
                "message": "Drug not found in FDA database. Consult a pharmacist or doctor."}

    def _first(field: str, maxlen: int = 400) -> str:
        val = data["results"][0].get(field, [])
        return (val[0] if val else "")[:maxlen] or "Not specified"

    return {
        "status":                "found",
        "drug_queried":          medication_name,
        "fda_generic_name":      search_term,
        "warnings":              _first("warnings"),
        "overdose":              _first("overdosage", 300),
        "contraindications":     _first("contraindications", 300),
        "warnings_and_cautions": _first("warnings_and_cautions", 300),
        "source":                "FDA_drug_label_openFDA",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3: Vital sign parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_vitals_from_text(text: str) -> dict:
    """
    Extract vital sign numbers from the ASHA worker's free-text reply.

    Handles common formats: "38.5°C", "102 F", "SpO2 94%", "pulse 88", "BP 110/70".
    Converts Fahrenheit to Celsius automatically.
    Any vital not mentioned returns as null.
    The output feeds directly into calculate_india_news2().

    Args:
        text: Free-text reply from the ASHA worker with measurement readings.
    """
    t = text.lower()

    # Temperature — requires a temp keyword OR an explicit unit. A bare number
    # is never treated as temperature ("BP 94" must not become 94°F → 34.4°C).
    temp_c = None
    m = re.search(
        r"(?:temp(?:erature)?|fever)[^0-9]{0,15}(\d{2,3}(?:\.\d)?)\s*°?\s*([fc]|celsius|fahrenheit)?", t
    )
    # Guard: the keyword→number gap must not cross another vital's keyword
    # ("no fever, pulse 76" must not become temp=76°F).
    if m and re.search(r"pulse|heart|bp|blood|oxygen|spo2|breath|resp|saturat", m.group(0)):
        m = None
    # Guard: ambient temperature is not a vital ("room temperature is 30 degrees").
    if m and re.search(r"(?:room|water|outside|weather|ambient)\s+temp", t[: m.start() + 12]):
        m = None
    if m is None:
        m = re.search(
            r"(\d{2,3}(?:\.\d)?)\s*(?:°\s*([fc])?|(?:degrees?\s*)?([fc]\b|celsius|fahrenheit))", t
        )
    if m:
        val  = float(m.group(1))
        unit = next((g for g in m.groups()[1:] if g), "").strip()
        if unit in ("f", "fahrenheit") or (val > 45 and unit not in ("c", "celsius")):
            temp_c = round((val - 32) * 5 / 9, 1)
        elif 30 <= val <= 43:
            temp_c = val

    # SpO2 / oxygen saturation
    spo2 = None
    for pat in (
        r"(?:oxygen|spo2|saturation|pulse\s*ox(?:imeter)?)[^0-9]{0,12}(\d{2,3})\s*(?:%|percent)?",
        r"(\d{2,3})\s*(?:%|percent)\s*(?:oxygen|saturation|spo2)",
        r"\b(\d{2})\s*(?:%|percent)",
    ):
        m = re.search(pat, t)
        if m:
            # A bare percentage is only SpO2 if it isn't humidity/battery/etc.
            if re.search(r"(?:humidity|battery|charge|chance)[^0-9]{0,12}$", t[: m.start()]):
                continue
            val = int(m.group(1))
            if 50 <= val <= 100:
                spo2 = val
                break

    # Heart rate / pulse — wider gap for phrases like "pulse is very fast
    # around 118", guarded so the gap never crosses another vital's keyword.
    hr = None
    m = re.search(r"(?:pulse|heart\s*rate|bpm|hr\b|beats?\s*(?:per\s*min)?)[^0-9]{0,25}(\d{2,3})", t)
    if m and not re.search(r"bp|blood|temp|oxygen|spo2|resp|breath|saturat", m.group(0)):
        val = int(m.group(1))
        if 30 <= val <= 250:
            hr = val

    # Respiratory rate — wider gap for "breathing fast, about 24 times",
    # guarded against crossing another vital's keyword.
    rr = None
    m = re.search(
        r"(?:breath(?:ing)?|respir(?:atory)?(?:\s*rate)?|breaths?\s*(?:per\s*min)?|rr\b)[^0-9]{0,20}(\d{1,3})", t
    )
    if m and not re.search(r"bp|blood|temp|oxygen|spo2|pulse|heart|saturat", m.group(0)):
        val = int(m.group(1))
        if 5 <= val <= 60:
            rr = val

    # Systolic blood pressure — keyword before the number ("BP machine shows 88")
    # or "systolic" after it ("88 systolic"); also bare "110/70" format.
    sbp = None
    m = (
        re.search(r"(?:b\.?p\.?|blood\s*pressure|systolic)[^0-9]{0,20}(\d{2,3})(?:\s*/\s*\d+)?", t)
        or re.search(r"(\d{2,3})\s*(?:systolic|\s*/\s*\d{2,3})", t)
    )
    if m:
        val = int(m.group(1))
        if 50 <= val <= 280:
            sbp = val

    # Consciousness level
    consciousness = "Alert"
    if any(w in t for w in ("unconscious", "not waking", "won't wake", "cannot wake")):
        consciousness = "Unresponsive"
    elif any(w in t for w in ("confused", "disoriented", "not responding to questions")):
        consciousness = "Voice"
    elif any(w in t for w in ("drowsy", "very sleepy", "lethargic", "hard to wake")):
        consciousness = "Pain"

    n = sum(1 for v in [temp_c, spo2, hr, rr, sbp] if v is not None)
    return {
        "status":            "extracted" if n > 0 else "no_vitals_found",
        "temp_c":            temp_c,
        "spo2":              spo2,
        "heart_rate":        hr,
        "resp_rate":         rr,
        "systolic_bp":       sbp,
        "consciousness":     consciousness,
        "n_vitals_extracted": n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4: India-adapted NEWS2 clinical severity score
# ─────────────────────────────────────────────────────────────────────────────

def calculate_india_news2(
    resp_rate:    Optional[int]   = None,
    spo2:         Optional[int]   = None,
    temp_c:       Optional[float] = None,
    systolic_bp:  Optional[int]   = None,
    heart_rate:   Optional[int]   = None,
    consciousness: str            = "Alert",
    context_flags: Optional[list] = None,
) -> dict:
    """
    Calculate the National Early Warning Score 2 (NEWS2), adapted for India.

    NEWS2 is a validated clinical deterioration score (Royal College of Physicians
    UK, 2017). Validated in Indian hospitals (PMC8673675, PMC11942526).

    India-specific additions:
    - SpO2 = 94% → MOHFW ASHA training advisory
    - Fever ≥38.5°C + monsoon/endemic context → +1 (malaria/dengue risk)
    - Temp >40°C + declared heat wave → ER flag (heat stroke)
    - Pregnant or elderly → escalate one level

    Args:
        resp_rate:    Breaths per minute (normal 12–20).
        spo2:         Oxygen saturation % (normal ≥95%).
        temp_c:       Temperature in Celsius (normal 36.1–37.2).
        systolic_bp:  Systolic BP in mmHg (normal 110–130).
        heart_rate:   Beats per minute (normal 60–100).
        consciousness: 'Alert', 'Voice', 'Pain', or 'Unresponsive'.
        context_flags: List of India modifiers. Options:
            'monsoon', 'endemic_malaria', 'heat_wave', 'pregnant', 'elderly', 'known_anemia'
    """
    flags   = [f.lower() for f in (context_flags or [])]
    score   = 0
    clinical_flags: list = []
    india_mods:     list = []

    # Respiratory rate
    if resp_rate is not None:
        if resp_rate <= 8 or resp_rate >= 25:
            score += 3; clinical_flags.append(f"resp_rate_{resp_rate}_critical")
        elif resp_rate <= 11 or resp_rate >= 21:
            score += 2
        elif resp_rate >= 9:
            score += 1

    # SpO2
    if spo2 is not None:
        if spo2 <= 91:
            score += 3; clinical_flags.append(f"spo2_{spo2}_critical")
        elif spo2 <= 93:
            score += 2; clinical_flags.append(f"spo2_{spo2}_very_low")
        elif spo2 <= 95:
            score += 1
        if spo2 == 94:
            india_mods.append(f"india_spo2_advisory: {spo2}% — MOHFW threshold → refer to DOCTOR")

    # Temperature
    if temp_c is not None:
        if temp_c <= 35.0:
            score += 3; clinical_flags.append("hypothermia")
        elif temp_c >= 39.1:
            score += 2; clinical_flags.append(f"high_fever_{temp_c}c")
        elif temp_c >= 38.1:
            score += 1; clinical_flags.append(f"fever_{temp_c}c")
        elif temp_c <= 36.0:
            score += 1
        if temp_c >= 38.5 and ("monsoon" in flags or "endemic_malaria" in flags):
            score += 1
            india_mods.append(f"india_endemic_fever: {temp_c}°C in endemic context → +1")
        if temp_c > 40.0 and "heat_wave" in flags:
            clinical_flags.append("india_heat_stroke_risk")
            india_mods.append(f"india_heat_wave: {temp_c}°C in heat wave → heat stroke risk")

    # Systolic BP
    if systolic_bp is not None:
        if systolic_bp <= 90 or systolic_bp >= 220:
            score += 3; clinical_flags.append(f"bp_{systolic_bp}_critical")
        elif systolic_bp <= 100:
            score += 2
        elif systolic_bp <= 110:
            score += 1

    # Heart rate
    if heart_rate is not None:
        if heart_rate <= 40 or heart_rate >= 131:
            score += 3; clinical_flags.append(f"hr_{heart_rate}_critical")
        elif heart_rate <= 50 or heart_rate >= 111:
            score += 2
        elif (41 <= heart_rate <= 50) or (91 <= heart_rate <= 110):
            score += 1

    # Consciousness
    if consciousness and consciousness != "Alert":
        score += 3; clinical_flags.append(f"consciousness_{consciousness.lower()}")

    # Vulnerable patient bump
    bump = False
    if "pregnant" in flags:
        india_mods.append("india_pregnant: escalate one level"); bump = True
    if "elderly" in flags:
        india_mods.append("india_elderly_≥70: escalate one level"); bump = True
    if "known_anemia" in flags and spo2 is not None and spo2 >= 92:
        india_mods.append("india_anemia: SpO2 may be falsely normal — verify clinically")

    # Escalation thresholds (standard NEWS2)
    if score >= 7 or "india_heat_stroke_risk" in clinical_flags:
        escalation = "ER";     risk = f"CRITICAL (score {score})"
    elif score >= 5:
        escalation = "ER";     risk = f"HIGH (score {score})"
    elif score >= 3:
        escalation = "DOCTOR"; risk = f"MEDIUM (score {score})"
    elif score >= 1:
        escalation = "DOCTOR"; risk = f"LOW-MEDIUM (score {score})"
    else:
        escalation = "WAIT";   risk = f"LOW (score {score})"

    # Systolic BP <=90 = hemodynamic compromise — this alone mandates ER
    # regardless of the aggregate score, because hypotension at this level
    # indicates shock physiology. NEWS2 score-4 would otherwise map to DOCTOR.
    if systolic_bp is not None and systolic_bp <= 90 and escalation != "ER":
        escalation = "ER"
        clinical_flags.append("bp_hemodynamic_compromise_er_override")

    if bump:
        if escalation == "WAIT":   escalation = "DOCTOR"
        elif escalation == "DOCTOR": escalation = "ER"

    n_vitals = sum(1 for v in [resp_rate, spo2, temp_c, systolic_bp, heart_rate] if v is not None)
    confidence = "HIGH" if n_vitals >= 4 else "MEDIUM" if n_vitals >= 2 else "LOW (limited vitals)"

    return {
        "news2_score":             score,
        "risk_level":              risk,
        "recommended_escalation":  escalation,
        "clinical_flags":          clinical_flags,
        "india_modifiers_applied": india_mods,
        "vitals_assessed":         n_vitals,
        "score_confidence":        confidence,
        "scoring_standard":        "NEWS2_RCP_UK_2017_India_adapted",
    }


# -----------------------------------------------------------------------------
# Medication caution (relay-only) -- surfaces lookup_drug_safety in the RESPONSE
# -----------------------------------------------------------------------------
# The pipeline detects a named medicine, looks up its safety profile, and the
# harness injects ONE plain-language caution line into the worker-facing
# response. This is RELAY-ONLY: it never prescribes, never changes a dose --
# it just makes sure a clinically relevant medicine is flagged to the doctor.

_COMMON_GENERICS = {
    "warfarin", "paracetamol", "acetaminophen", "ibuprofen", "aspirin",
    "insulin", "amoxicillin", "azithromycin", "metformin", "diclofenac",
}


def medication_caution(patient_input):
    """Detect a medicine named in the free text and return a relay-only caution.

    Returns (caution_line: str | None, drug_safety: dict | None).
    Detection is deterministic (scans for known Indian brand/generic names), so
    it does not depend on the LLM choosing to call the tool. Never prescribes.
    """
    text = (patient_input or "").lower()
    candidates = set(INDIA_DRUG_ALIASES.keys()) | _COMMON_GENERICS
    found = None
    for name in sorted(candidates, key=len, reverse=True):
        if re.search(r"\b" + re.escape(name) + r"\b", text):
            found = name
            break
    if not found:
        return None, None

    ds = lookup_drug_safety(found)
    warn = str(ds.get("warnings", "") or "").strip()
    drug_disp = ds.get("fda_generic_name") or ds.get("drug_queried") or found

    short = ""
    if warn and warn.lower() not in ("not specified", "no data.", "no data"):
        short = re.split(r"(?<=[.])\s", warn)[0].strip()
        short = (short[:160] + "...") if len(short) > 160 else short

    if short:
        caution = (
            f"Medicine note: the patient is taking {drug_disp}. {short} "
            "Tell the doctor about this medicine; do not stop or change the dose on your own."
        )
    else:
        caution = (
            f"Medicine note: the patient is taking {drug_disp}. Tell the doctor they are "
            "on this medicine; do not change the dose without medical advice."
        )
    return caution, ds


def attach_medication_note(state, patient_input, disclaimer=None):
    """If a medicine is named, store its safety dict in state['drug_safety'] and
    inject the caution line into state['final_response'] just before the
    disclaimer (so the disclaimer stays last). Returns the same state dict."""
    caution, ds = medication_caution(patient_input)
    if not caution:
        return state
    state["drug_safety"] = ds
    final = str(state.get("final_response", "") or "")
    if disclaimer and disclaimer in final:
        state["final_response"] = final.replace(disclaimer, caution + "\n\n" + disclaimer)
    elif final.strip():
        state["final_response"] = final.rstrip() + "\n\n" + caution
    else:
        state["final_response"] = caution
    return state
