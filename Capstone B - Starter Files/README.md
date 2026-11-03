# Sahayak Health AI — Capstone Delivery Package

**Sahayak Health AI** (सहायक, "aid") — AI triage assistant for India's frontline health workers (ASHA/ANM).
Built on Google ADK + hermes3:8b (Ollama) / Gemini Flash.

## Folder structure

```
Sahayak_Capstone/
├── learner/          ← YOUR WORK: fill in the four notebooks + sahayak_starter.py
├── src/              ← provided Python modules (demo, eval, tools) — do not submit
├── data/             ← cases.db + raw JSONL dataset
├── tests/            ← pytest harness — run after `agent_pipeline_development.ipynb` to self-check
├── docs/             ← capstone documents (problem, workflow, task brief)
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
cd src && python setup_db.py           # creates cases.db (if not present)
jupyter notebook ../learner/adk_foundations.ipynb
```

> Notebooks use `sys.path.insert(0, '../src')` — run from the `learner/` directory,
> or add the `src/` path to your PYTHONPATH.

## Document index

| File | Purpose |
|------|---------|
| docs/LEARNER_TASK_BRIEF.md | **Start here** — what to build, notebook by notebook |
| docs/CAPSTONE_PROBLEM_STATEMENT.md | Problem context, dataset, evaluation criteria |

## Learner notebooks

| Notebook | Mode | What the learner does |
|----------|------|----------------------|
| adk_foundations.ipynb | Taught | Run + observe ADK primitives, answer 5 reflection questions |
| data_understanding_and_baseline.ipynb | Scaffolded | EDA, triage mapping, policy baseline, FunctionTools exhibit |
| agent_pipeline_development.ipynb | Semi-scaffolded | Write 6 agent instructions, wire pipeline, run 20-case eval |
| agent_evaluation_and_optimisation.ipynb | Open-ended | 50-case eval, failure analysis, implement 1 improvement, demo |

The `tests/` suite (12 tests) runs from the package root via `pytest tests/`.
Run it after `agent_pipeline_development.ipynb` to verify your safety evaluator implementation.
