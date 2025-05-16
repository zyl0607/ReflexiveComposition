"""
Temporal QA Case Study — Reflexive Composition Framework

This script demonstrates how grounding with a structured knowledge graph can improve
temporal question-answering, using the recent 2025 EFL Cup victory by Newcastle United.
"""

import os
import logging
from reflexive_composition.core import ReflexiveComposition
from reflexive_composition.hitl.interface import ConsoleValidationInterface
from reflexive_composition.knowledge_graph.graph import KnowledgeGraph
from reflexive_composition.utils.llm_utils import log_result
from prompt_templates import WITH_CONTEXT_PROMPT_TEMPLATE, NO_CONTEXT_PROMPT_TEMPLATE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    print("\n=== Temporal QA Case Study: Newcastle United 2025 EFL Cup ===\n")

    # --- Configuration ---
    kb_llm_config = {
        "model_name": os.environ.get("KB_LLM_MODEL", "gemini-2.0-flash"),
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "model_provider": "google",
    }

    target_llm_config = {
        "model_name": os.environ.get("TARGET_LLM_MODEL", "gemini-2.0-flash"),
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "model_provider": "google",
    }

    kg_config = {
        "storage_type": "in_memory",
        "schema": {
            "entity_types": ["Team", "Trophy", "Match", "Player", "Date"],
            "relationship_types": ["Won", "PlayedIn", "Scored", "OccurredOn", "EndedDrought"],
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
    # Optional: Load from structured data instead of free text
    json_path = os.path.join("data", "newcastle_efl_cup_2025.json")
    if os.path.exists(json_path):
        import json
        with open(json_path) as f:
            structured_data = json.load(f)
        print("\nLoaded structured data:")
        for rel in structured_data["relations"]:
            print(" -", rel["subject"], rel["predicate"], rel["object"])
        print("\n(You can switch to use this KG directly for QA if needed.)\n")
    source_text = """
    Newcastle United defeated Liverpool 2–1 in the 2025 EFL Cup final held at Wembley on March 16, 2025.
    It was Newcastle’s first major domestic trophy in 70 years, having last won the FA Cup in 1955.
    Goals were scored by Alexander Isak and Bruno Guimarães.
    """

    print("Extracting knowledge from match summary...\n")
    extraction_result = rc.extract_knowledge(
        source_text=source_text,
        schema=kg_config["schema"],
        confidence_threshold=0.7
    )

    rc.knowledge_graph.add_triples(extraction_result["triples"])
    
    print("\nExtracted Triples:")
    for triple in extraction_result["triples"]:
        print(" -", triple)

    print("\nAnswering temporal question without knowledge...\n")
    prompt_result = rc.generate_response(
        query="When did Newcastle United last win a major domestic trophy?",
        grounded=False,
        template=NO_CONTEXT_PROMPT_TEMPLATE,
    )

    print("Generated Answer:")
    print(prompt_result)

    log_result(prompt_result)
    print("\nAnswering temporal question with knowledge...\n")
    prompt_result = rc.generate_response(
        query="When did Newcastle United last win a major domestic trophy?",
        grounded=True,
        template=WITH_CONTEXT_PROMPT_TEMPLATE,
    )

    print("Generated Answer:")
    print(prompt_result)
    log_result(prompt_result)


if __name__ == "__main__":
    main()
