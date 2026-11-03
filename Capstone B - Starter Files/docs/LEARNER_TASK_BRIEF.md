# Sahayak Health AI Capstone - Learner Task Brief

## What You Are Building

You will build **Sahayak Health AI**, a Google ADK agent system that supports an
ASHA/frontline health worker in deciding one of three care levels from a
patient's symptom description:

- `WAIT`: manage at home / routine care for now
- `DOCTOR`: see a doctor or clinic soon
- `ER`: go now to the nearest hospital / CHC / PHC or call 108

The agent is **decision support only**. It must not diagnose, prescribe, or
replace a qualified medical professional.

## What You Submit

Your submission is notebook-first:

| Notebook | Marks | Submit / complete |
|---|---:|---|
| ADK Foundations | 9 | `adk_foundations.ipynb` concept answers |
| Data Understanding & Baseline | 21 | `data_understanding_and_baseline.ipynb` and the corresponding functions in `sahayak_starter.py` |
| Agent Pipeline Development | 29 | `agent_pipeline_development.ipynb` with your ADK agent instructions and pipeline assembly |
| Agent Evaluation & Optimisation | 41 | `agent_evaluation_and_optimisation.ipynb` (evaluation, safety evaluator, failure analysis, improvement log) **plus `final_report.pdf`** |
| **Total** | **100** | |

Your six graded deliverables are the **four notebooks**, **`sahayak_starter.py`**, and a **`final_report.pdf`**. The required sections and marks for the final report are listed in the final submission checkpoint of **`agent_evaluation_and_optimisation.ipynb`**. Files in `src/` are provided as support modules (demo, evaluation, tools, etc.) and **should not be submitted**.

## Notebook 1 - Learn ADK Basics

You will run small ADK examples and answer concept questions.

You must show that you understand:

- `LlmAgent`
- `SequentialAgent`
- `Runner`
- `session.state`
- `output_key`
- how one agent's output becomes the next agent's input

## Notebook 2 - Dataset, EDA, And Policy Baseline

You will work with `gretelai/symptom_to_diagnosis`, map diagnosis labels to
`WAIT` / `DOCTOR` / `ER`, and build a deterministic baseline.

You must complete:

- dataset checks and EDA notes
- diagnosis-to-triage mapping review
- locked evaluation sample: `n=50, seed=42`
- `score_severity()`
- `decide_triage()`
- `run_policy_triage()`
- baseline metrics and failure examples
- a short system-design note

## Notebook 3 - Build The ADK Agent

You will build the agent pipeline:

```text
symptom_parser -> severity_scorer -> followup_asker -> triage_decider -> response_formatter
```

The `safety_evaluator` runs after the pipeline as an audit.

Your `triage_decider` must keep all four tools available:

- vitals extraction
- NEWS2 scoring
- case-memory database search
- drug safety lookup

The database/RAG tool is evidence for the agent. It is not the grading label
source.

## Notebook 4 - Evaluate, Diagnose, Improve

You will evaluate the agent like a safety-critical system.

You must complete:

- policy vs ADK comparison
- confusion matrix
- per-class recall
- under-triage count and rate
- deterministic `safety_evaluator_agent()`
- evaluator pass rate and human-review rate
- top three failure patterns with failing stage named
- one measured improvement with before/after metrics

## Safety Rules

Your final response must:

- give exactly one care level: `WAIT`, `DOCTOR`, or `ER`
- use India-specific emergency guidance: nearest hospital / CHC / PHC or call 108
- include the required disclaimer
- avoid diagnosis language
- avoid prescription language
- escalate red flags; never send a true emergency home

## How You Are Graded

The grader will look for evidence in your notebooks and outputs. High accuracy
alone is not enough. In this capstone, **ER recall, under-triage, and safety
compliance matter more than raw accuracy**.

If your reported score is unusually high, the grader will check for leakage,
fallback-dataset use, or copied solution outputs.
