# Reflexive Composition Case Studies

This directory contains case studies demonstrating the application of the **Reflexive Composition** framework across different domains:

- `basic_example.py`: A didactic illustration of the full Reflexive Composition pipeline. The content previously discussed in this README primarily refers to this example.
- `code_security/`: Applies Reflexive Composition to detect and mitigate deprecated or insecure API usage in code generation.
- `temporal_qa/`: Being refactored to model recency-sensitive question answering using recent events (e.g., 2025 EFL Cup win).
- `medical_data/`: *(Coming soon)* Use of Reflexive Composition for extracting and grounding private clinical knowledge.

---

# Temporal QA Benchmark

This benchmark demonstrates how structured knowledge can mitigate brittleness in language models when answering time-sensitive questions. The core goal is to compare outputs of models on temporally anchored facts, such as recent sports events or political developments.

## Dataset Overview

The benchmark contains a set of factual assertions and multiple phrased questions relating to each fact, such as:

- “When did [X] last win [Y]?”
- “Has [X] ever happened before [Z]?”
- “What is the most recent event involving [X]?”

These are paired with a reference knowledge graph and corresponding fact timestamp.

## Objective

To test:

- If language models respond with stale, outdated, or hallucinated information.
- Whether grounding them with an LLM-generated, validated KG improves factual accuracy and temporal precision.

## Example Use Case

The current benchmark includes data related to past examples such as the Trump shooting in July 2024. We aim to replace this with a newer, less sensitive example like Newcastle United’s 2025 EFL Cup victory for long-term clarity and neutrality.

## Running the Pipeline

You can execute the pipeline with a basic example via:

```bash
python basic_example.py