import os
import json
from typing import Optional
from reflexive_composition.core import ReflexiveComposition
from reflexive_composition.utils.llm_utils import log_result
from prompt_templates import WITH_CONTEXT_TEMPLATE, NO_CONTEXT_TEMPLATE


def run_single_query(query: Optional[str] = None, grounded: bool = True, template: Optional[str] = None):
    print("\n=== Medical QA Case Study: FHIR + RxNorm Grounding ===\n")

    # --- Configuration ---
    kb_llm_config = {
        "model_name": os.environ.get("KB_LLM_MODEL", "gemini-2.0-flash"),
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "model_provider": "google",
    }

    target_llm_config = {
        "model_name": os.environ.get("KB_LLM_MODEL", "gemini-2.0-flash"),
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "model_provider": "google",
    }

    kg_config = {
        "storage_type": "in_memory",
        "schema": {
            "entity_types": ["Patient", "Medication", "Condition", "Allergy", "Code"],
            "relationship_types": ["HasCondition", "Prescribed", "ContraindicatedWith", "MappedTo"],
            "version": 1
        }
    }

    hitl_config = {
        "high_confidence_threshold": 0.85,
        "low_confidence_threshold": 0.6
    }

    # --- Initialize Reflexive Composition framework ---
    rc = ReflexiveComposition(
        kb_llm_config=kb_llm_config,
        target_llm_config=target_llm_config,
        kg_config=kg_config,
        hitl_config=hitl_config
    )

    # --- Source Text ---
    source_path = os.path.join(os.path.dirname(__file__), "data", "fhir_patient.json")
    with open(source_path, "r") as f:
        source_text = f.read()

    # --- Extract and validate knowledge ---
    extraction_result = rc.extract_knowledge(source_text)
    rc.knowledge_graph.add_triples(extraction_result["triples"])

    # --- Add RxNorm enrichment ---
    rx_path = os.path.join(os.path.dirname(__file__), "data", "synthetic_rxnorm_triples.json")
    if os.path.exists(rx_path):
        with open(rx_path, "r") as f:
            rx_triples = json.load(f)
            rc.knowledge_graph.add_triples(rx_triples)

    # --- User Query ---
    user_query = query or "Is this patient currently on a blood pressure medication?"

    if not template:
        template = WITH_CONTEXT_TEMPLATE if grounded else NO_CONTEXT_TEMPLATE

    # --- Generate response ---
    result = rc.generate_response(
        query=user_query,
        grounded=grounded,
        template=template,
        context_label="Context (extracted from clinical data)"
    )

    log_result({**result, "query": user_query})
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Single query string")
    parser.add_argument("--query-file", type=str, help="Path to .jsonl file of queries")
    parser.add_argument("--grounded", action="store_true", default=True, help="Use grounded context (default: True)")
    parser.add_argument("--no-grounded", dest="grounded", action="store_false", help="Disable grounding")
    args = parser.parse_args()

    if args.query_file:
        with open(args.query_file) as f:
            for line in f:
                q = json.loads(line)
                run_single_query(
                    query=q["query"],
                    grounded=args.grounded,
                    template=WITH_CONTEXT_TEMPLATE if args.grounded else NO_CONTEXT_TEMPLATE
                )
    else:
        run_single_query(
            query=args.query or "Is this patient currently on a blood pressure medication?",
            grounded=args.grounded,
            template=WITH_CONTEXT_TEMPLATE if args.grounded else NO_CONTEXT_TEMPLATE
        )
