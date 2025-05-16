"""
Code Security Case Study — Reflexive Composition Framework

This script demonstrates the full Reflexive Composition pipeline to analyze deprecated
or insecure APIs and generate safe code suggestions.
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
    print("\n=== Code Security Case Study ===\n")

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
            "entity_types": ["API", "Function", "Vulnerability", "Deprecation", "Version"],
            "relationship_types": ["DeprecatedIn", "ReplacedBy", "CausesVulnerability"],
            "version": 1
        }
    }

    hitl_config = {
        "high_confidence_threshold": 0.8,
        "low_confidence_threshold": 0.5
    }

    # --- Initialize Reflexive Composition framework ---
    rc = ReflexiveComposition(
        kb_llm_config=kb_llm_config,
        target_llm_config=target_llm_config,
        kg_config=kg_config,
        hitl_config=hitl_config
    )

    # --- Source Text for Extraction ---
    source_text = """
    The Python 'cgi' module was officially deprecated in version 3.11 due to long-standing
    security issues. Developers are advised to use safer alternatives like 'http.server'
    or third-party libraries. The 'cgi.escape' method, in particular, has caused XSS vulnerabilities
    in the past.
    """

    print("Extracting knowledge from code security input...\n")
    extraction_result = rc.extract_knowledge(
        source_text=source_text,
        schema=kg_config["schema"],
        confidence_threshold=0.7
    )

    rc.knowledge_graph.add_triples(extraction_result["triples"])

    print("\nExtracted Triples:")
    for triple in extraction_result["triples"]:
        print(" -", triple)

    print("\nGenerating secure code suggestion without knowledge...\n")
    prompt_result = rc.generate_response(
        query="What is a safe alternative to using the 'cgi.escape' function in modern Python?",
        grounded=False,
        template=NO_CONTEXT_PROMPT_TEMPLATE,
    )

    print("Generated Answer:")
    print(prompt_result)

    log_result(prompt_result)

    print("\nGenerating secure code suggestion with knowledge...\n")
    prompt_result = rc.generate_response(
        query="What is a safe alternative to using the 'cgi.escape' function in modern Python?",
        grounded=True,
        template=WITH_CONTEXT_PROMPT_TEMPLATE,
    )

    print("Generated Answer:")
    print(prompt_result)
    
    log_result(prompt_result)


if __name__ == "__main__":
    main()
