import os
from reflexive_composition.core import ReflexiveComposition
from eval_utils import run_query_loop
from examples.medical_data.prompt_templates import WITH_CONTEXT_TEMPLATE, NO_CONTEXT_TEMPLATE

# Initialize ReflexiveComposition
rc = ReflexiveComposition()

# Path to query file
queries_path = os.path.join("data", "medical_data_queries.jsonl")

# Run evaluation
run_query_loop(
    rc=rc,
    queries_path=queries_path,
    template_with_kg=WITH_CONTEXT_TEMPLATE,
    template_without_kg=NO_CONTEXT_TEMPLATE,
    context_label="Context (extracted from clinical data)",
    save_results="results_medical.jsonl"
)
