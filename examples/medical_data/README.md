# 🏥 Medical QA Case Study — FHIR and RxNorm Integration

This case study demonstrates how the **Reflexive Composition** framework can be used in a healthcare setting to extract, validate, and reason over structured patient data (FHIR-style) alongside external medical knowledge (RxNorm ontology).

## 🧬 Scenario

Given a synthetic patient's clinical summary, the system:

1. Extracts entities and relationships (conditions, medications) from a structured FHIR-style file
2. Enriches the patient-specific graph with external facts from a RxNorm mini-ontology
3. Answers clinical queries grounded in this multi-source knowledge graph

## 🔁 Reflexive Composition Flow

- **LLM2KG**: Converts FHIR JSON into triples (A-graph)
- **RxNorm Import**: Loads external class information (T-graph)
- **HITL Validation**: Applies optional confidence thresholds
- **KG2LLM**: Generates answers using both graphs as context

Example questions include:
- _"What class of drug is Metformin?"_
- _"Is the patient currently on a blood pressure medication?"_
- _"Is the patient taking a blood thinner?"_

## 📂 Files

- `run_medical.py` — CLI interface to run single or batched queries
- `data/`
  - `fhir_patient.json` — structured input for patient-specific facts
  - `synthetic_rxnorm_triples.json` — external ontology fragments
  - `medical_data_queries.jsonl` — reusable evaluation queries

## 🚀 How to Run

### Run a single grounded query:

```bash
python run_medical.py --query "What class of drug is Metformin?" --grounded
```

### Run multiple queries from file:

```bash
python run_medical.py --query-file data/medical_data_queries.jsonl --grounded
```

## ✅ Output

Each run logs:
- The extracted triples used
- The generated prompt
- Whether contradictions were detected
- The model’s response (grounded or ungrounded)

## 📌 Notes

- This implementation supports integration of patient-specific FHIR data (A-graph) and external RxNorm knowledge (T-graph) in a single prompt.
- LLM responses are intentionally restricted to structured knowledge to avoid hallucination.
