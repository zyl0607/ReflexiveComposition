# Temporal QA Case Study — Newcastle United 2025 EFL Cup Victory

This case study demonstrates how the **Reflexive Composition** framework improves time-sensitive question answering by grounding LLM output in structured knowledge graphs built from recent events.

## ⚽ Scenario: Newcastle Ends 70-Year Trophy Drought

In March 2025, Newcastle United defeated Liverpool in the EFL Cup final, securing their first major domestic trophy since 1955. This fresh event is used to evaluate how well models:
- Recognize and reason over recent events
- Avoid outdated or hallucinated answers

## 🔁 Reflexive Composition Flow

This case study follows the Reflexive Composition framework:

1. **LLM2KG (Knowledge Extraction)**:
   - Structured triples are extracted from a match report paragraph.

2. **HITL Validation**:
   - Confidence thresholds or human review ensure accurate KG contents.

3. **KG2LLM (Grounded Generation)**:
   - The validated knowledge graph is used to answer:
     _“When did Newcastle United last win a major domestic trophy?”_

## 📁 Files

- `run_temporal.py` — Main runner script for the experiment
- `data/newcastle_efl_cup_2025.json` — (Optional) Structured source facts
- `results/` — Placeholder for outputs from different LLMs

## 🚀 How to Run

```bash
python run_temporal.py