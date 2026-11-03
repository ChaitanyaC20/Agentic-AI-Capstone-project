# Capstone: Sahayak Health AI - Build a Medical Triage Agent
> **Sahayak Health AI** (सहायक, "aid") - an AI triage assistant for India's frontline health workers
### AI Agents with Google ADK |Capstone

## The Story

Priya is an ASHA worker in rural Rajasthan. Villagers come to her with symptoms,
but the nearest doctor may be many kilometres away. The practical question is not
"what disease is this?" The practical question is:

> Should this person wait, see a doctor today, or go to emergency care now?

Your capstone is to build **Sahayak Health AI**, an ADK-based decision-support agent
that helps Priya make that next-step recommendation safely.

## What The Agent Must Do

- Read a short symptom description.
- Extract the symptoms that are actually visible in the text.
- Ask one clarifying question only when the case is ambiguous — **and wait for
  the answer**. The worker's answer feeds the triage decision (a real
  multi-turn loop, not a decorative question).
- Recommend exactly one level: `WAIT`, `DOCTOR`, or `ER`.
- Explain the reason in calm, plain language.
- Never diagnose, prescribe, or pretend to replace a clinician.

Why the follow-up loop matters: real diagnosis needs information the first
message never contains — history, vitals, labs, imaging. A triage agent cannot
collect blood or order an X-ray, so its core skill is knowing **what
information it lacks**: ask for what a health worker can observe, and escalate
when the missing information is the kind only a clinic can provide.

Required disclaimer:

```text
This is decision support guidance only. Always consult a qualified medical professional for diagnosis and treatment.
```

## Why ADK

We use Google Agent Development Kit because the learner should experience a
real agent framework without drowning in framework code. ADK gives us:

- `LlmAgent` for one focused reasoning step.
- `SequentialAgent` for a visible multi-step workflow.
- Session state for passing outputs between agents.
- Runner/session services for execution and traceable experiments.
- Built-in evaluation concepts for trajectory, response quality, and safety.

The capstone is not about building the most complex agent. It is about building a
system that can grow: baseline first, evaluation next, improvement after evidence.

## Industry And Research Anchors

This project is inspired by three current signals:

- Google ADK: production-oriented agent framework for building, debugging,
  evaluating, and deploying agents.
- Google MedGemma Impact Challenge on Kaggle: health-AI prototypes were judged
  on real-world clinical usefulness and human-centred design.
- AMIE research from Google Research/DeepMind: conversational medical AI needs
  evaluation across history-taking, management reasoning, communication, safety,
  and empathy, not only final-answer accuracy.

Readings are listed at the end of this brief.

## Dataset

Primary dataset: `gretelai/symptom_to_diagnosis` (Apache 2.0).

- **853 train cases / 212 test cases**, 22 diagnosis labels.
- `input_text`: a first-person, natural-language symptom description
  ("I've been having a lot of pain in my neck and back...") — realistic
  messy input, exactly what an ASHA worker would relay.
- `output_text`: the diagnosis label, mapped by the provided `TRIAGE_MAP`
  into care-seeking labels:

- `WAIT`: low urgency
- `DOCTOR`: clinical review needed
- `ER`: emergency care now

The **train split** feeds notebooks, the policy baseline, and the DSPy dev
sample. The **test split** is reserved for the live-agent evaluation
(`eval_agent.py`) and is never used for prompt tuning.

The diagnosis label is used only to build evaluation ground truth. The agent must
not output a diagnosis to the patient.

## ADK Architecture — Two-Phase Interactive Pipeline

```text
PHASE A — intake
patient_input
    |
    v
symptom_parser      -> symptoms
    |
    v
severity_scorer     -> severity_json
    |
    v
followup_asker      -> followup (question for the worker)
    |
    v
[PAUSE — the ASHA worker answers the question]
    |
    v
PHASE B — decision
triage_decider      -> triage_decision  (reads the answer; may escalate)
    |
    v
response_formatter  -> final_response
    |
    v
safety_evaluator    -> safety_audit  (LLM auditor, cross-checked by a
                                      deterministic rule judge)
```

Batch evaluation runs the same pipeline unattended (`auto` mode) — the decider
then falls back to the base severity rule when no answer is available.

Each stage has one job. This keeps the system debuggable. If the final answer is
wrong, learners can inspect whether the failure came from extraction, severity,
follow-up logic, triage rules, final communication, or evaluator audit.

## Clinical Depth Added To The Teaching Version

Sahayak now includes a clinical-safety layer while staying beginner-friendly:

- case-memory retrieval through `cases.db` and the
  `search_symptom_cases_db()` FunctionTool, used as similar-case evidence while
  keeping `gretelai/symptom_to_diagnosis` as the graded ground-truth dataset
- red-flag groups: breathing/circulation, neurology, infection/dehydration, metabolic
- optional vitals: oxygen saturation, temperature, heart rate, systolic blood pressure
- vulnerability cues: baby, child, elderly, pregnancy, known risk conditions
- guideline-style safety notes: why `WAIT`, `DOCTOR`, or `ER` is recommended
- under-triage review: cases where the agent is less urgent than the reference label
- evaluator agent: audits whether the final output followed all safety directions

This is the right level of clinical depth for the capstone. Learners see how a
health agent becomes safer without needing to run a large medical model.

## Harness Engineering

The capstone includes a safety harness around the agent:

- fixed evaluation sample
- deterministic metric computation
- evaluator verdict: `PASS` or `FLAG`
- violation list
- human-review flag
- stage-to-debug label

The evaluator checks strict compliance: no diagnosis, no prescription, required
disclaimer, valid triage label, no red-flag under-triage, and explanation aligned
with symptoms, vitals, and triage.

**The judge never grades itself.** Reported safety metrics come from the
deterministic rule evaluator (`safety_evaluator_agent` — regex + rules), not
from the LLM auditing its own output. The stage-6 LLM auditor is kept as a
teaching device, and its agreement with the deterministic judge is itself a
reported metric.

## Production Gate — DSPy Prompt Optimization

The capstone closes with a measured optimization pass: DSPy **MIPROv2**
compiles the triage-decision prompt on 20 hand-authored train cases and is
evaluated before/after on a **held-out dev set** sampled from the real dataset
(never shown to the optimizer). The gate criterion is a clinically graded loss
matrix (ER→WAIT = 10, DOCTOR→WAIT = 5, ER→DOCTOR = 3, over-triage = 1): the
weighted clinical cost must improve and the sent-home-in-error count must not
worsen. Every run writes `dspy_gate_results.json` with per-case provenance, and
the trust dashboard renders the artifact as the Production Readiness Gate.

## Notebooks Flow

### Notebook 1 - Learn ADK Through Small Examples

Learners already know GenAI basics. This notebook moves them from prompts to systems:

- Why agents need a system design.
- What an ADK `LlmAgent` is.
- How `SequentialAgent` chains specialised agents.
- How state and traces make debugging possible.
- Why evaluation comes before "clever improvements."

Deliverable: small ADK demo notebook and one-page design sketch.

### Notebook 2 - Understand The Health Problem And Build The Baseline

- Load the symptom dataset.
- Inspect `.head()` first, then distributions.
- Build the diagnosis-to-triage mapping.
- Create a fixed evaluation sample.
- Build the first two ADK stages: parser and severity scorer.
- Create a trace table.

Deliverable: EDA, mapping table, evaluation sample, two-stage baseline.

### Notebook 3 - Build The Full Agent

- Add follow-up question generation — and close the loop: collect the answer
  and pass it to the triage decision.
- Add triage decision (escalates on a worrying answer, never de-escalates).
- Add safe response formatting.
- Add the safety evaluator.
- Run one `WAIT`, one `DOCTOR`, and one `ER` demo, plus one case where the
  follow-up answer changes the decision.
- Run the fixed evaluation set again.

Deliverable: working ADK `SequentialAgent` and traceable outputs.

### Notebook 4 - Evaluate, Calibrate, And Present

- Compute under-triage rate, over-triage rate, accuracy, and per-level recall.
- Inspect the confusion matrix and identify the dominant error mode.
- **Calibrate the severity rubric on the train split**: the v1 rubric escalates
  on symptom drama ("severe headache with vomiting" sounds urgent but is the
  migraine pattern — a WAIT). Rewrite the rubric against the dataset's actual
  condition archetypes and A/B it on a train sample. This is the core 
  lesson: *pain intensity is not urgency.*
- Judge results against literature-anchored thresholds, not gut feel:
  under-triage ≤ 5% (ACS field-triage standard), over-triage ≤ 50%,
  accuracy ≥ 60% (a fine-tuned Llama-3.1-8B reaches 58.4% on real ESI
  triage; human nurses agree with each other only 74% of the time on the
  Manchester Triage System).
- **Measure the follow-up question, don't just demo it**: policy compliance
  (the agent asks exactly when severity is 2–3 — an emergency never waits on
  a question), relevance (the question is anchored to a symptom or red flag),
  and the loop-closure probe (a red-flag answer escalates the decision, a
  missing answer never lowers it).
- Run the held-out test sample once, after all tuning is frozen.

Deliverable: final notebook, evaluation report with acceptance table,
failure analysis, A/B calibration table, demo.

## What Good Looks Like

A good project is not a perfect medical system. A good project is:

- Honest about scope.
- Easy to inspect.
- Conservative on dangerous symptoms.
- Clear about failures.
- Measured on the same evaluation sample before and after improvement.
- Kind in language without pretending to be a doctor.

## How This Compares To MedGemma Challenge Work

Sahayak is a foundation capstone. The MedGemma Impact Challenge winners are
closer to product-grade health AI prototypes. They often use medical foundation
models, multimodal inputs, offline deployment, and guideline grounding. Sahayak
teaches the same design direction at a learner-safe level:

| Capability | Sahayak capstone | MedGemma challenge winners |
|---|---|---|
| Main goal | Teach agent design and evaluation | Build health AI prototypes |
| Inputs | symptom text, optional vitals | text, images, audio, X-rays, pathology |
| Model depth | Gemini/ADK plus transparent rules | MedGemma, MedSigLIP, HeAR, MedASR |
| Clinical grounding | red flags and care-level policy | WHO/MSF/OpenMRS or specialist workflows |
| Deployment | Colab-friendly teaching package | edge/offline/product demos |
| Evaluation | accuracy, recall, safety, failures | clinical usefulness and prototype impact |

The distinction path is to make Sahayak more like those systems by adding
guideline-grounded RAG, local-language intake, vitals, or a MedGemma reading
comparison.

## Optional Advanced Architecture: Parallel Clinical Review

The core learner path stays sequential because it is easier to understand. For a
distinction path, Sahayak can add one parallel block:

```text
symptom_parser
    |
    v
parallel_clinical_review
    |-- red_flag_reviewer
    |-- vitals_reviewer
    |-- guideline_reviewer
    v
triage_synthesizer
    |
    v
response_formatter
    |
    v
safety_evaluator
```

This is useful because red flags, vitals, and guideline notes are independent
checks. They can run at the same time, then one accountable decision node
synthesizes them. Do not parallelize the final triage decision itself. Healthcare
workflows need one final recommendation, not three competing answers.

## Suggested Readings

- Google ADK docs: https://adk.dev/
- ADK tutorials: https://google.github.io/adk-docs/tutorials/
- ADK evaluation docs: https://google.github.io/adk-docs/evaluate/
- ADK safety docs: https://google.github.io/adk-docs/safety/
- Google MedGemma Impact Challenge winners: https://blog.google/innovation-and-ai/technology/health/med-gemma-impact-challenge/
- Health AI Developer Foundations overview: https://developers.google.com/health-ai-developer-foundations/overview
- AMIE paper: https://research.google/pubs/towards-conversational-diagnostic-ai/
