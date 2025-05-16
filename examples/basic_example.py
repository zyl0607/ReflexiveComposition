# examples/basic_example.py
"""
Basic example demonstrating the Reflexive Composition framework.

This example shows the end-to-end flow of the Reflexive Composition framework,
including knowledge extraction, validation, and enhanced generation.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from reflexive_composition.utils.llm_utils import log_result

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import Reflexive Composition
from reflexive_composition.core import ReflexiveComposition
from reflexive_composition.hitl.interface import ConsoleValidationInterface
from reflexive_composition.knowledge_graph.graph import KnowledgeGraph

def main():
    """Run the basic example."""
    print("=== Reflexive Composition Basic Example ===\n")
    
    # Set up configuration
    kb_llm_config = {
        "model_name": os.environ.get("KB_LLM_MODEL", "gemini-2.0-flash"),
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "model_provider": "google",
    }
    
    target_llm_config = {
        "model_name": os.environ.get("TARGET_LLM_MODEL", "gemini-2.0-flash"),
        "api_key": os.environ.get("GOOGLE_API_KEY"),
        "model_provider": "google"
    }
    
    kg_config = {
        "storage_type": "in_memory",
        "schema": {
            "entity_types": ["Person", "Event", "Location", "Organization"],
            "relationship_types": ["LocatedIn", "WorksAt", "AttendedBy", "InvolvedIn"],
            "version": 1
        }
    }
    
    hitl_config = {
        "high_confidence_threshold": 0.8,
        "low_confidence_threshold": 0.5
    }
    
    # Initialize Reflexive Composition framework
    print("Initializing Reflexive Composition framework...")
    rc = ReflexiveComposition(
        kb_llm_config=kb_llm_config,
        target_llm_config=target_llm_config,
        kg_config=kg_config,
        hitl_config=hitl_config
    )
    print("Framework initialized!\n")
    
    # Example source text for knowledge extraction
    source_text = """
    Donald Trump survived an assassination attempt during a rally in Butler, Pennsylvania on July 13, 2024.
    He was grazed on the right ear by a bullet fired by a 20-year-old man, who was killed by Secret Service agents.
    The rally was attended by thousands of supporters. Trump continued his campaign after the incident.
    """
    
    print("=== Step 1: Knowledge Extraction ===")
    print("Extracting knowledge from source text...")
    print(f"Source text: {source_text}\n")
    
    extraction_result = rc.extract_knowledge(
        source_text=source_text,
        schema=kg_config["schema"],
        confidence_threshold=0.7  # Auto-accept triples with confidence >= 0.7
    )
    
    print(f"Extracted {len(extraction_result['triples'])} triples")
    for i, triple in enumerate(extraction_result['triples'], 1):
        print(f"Triple {i}: {triple.get('subject')} - {triple.get('predicate')} - {triple.get('object')}")
    print()
    
    # Update knowledge graph with extracted triples
    print("=== Step 2: Knowledge Graph Update ===")
    print("Updating knowledge graph with extracted triples...")
    
    update_success = rc.update_knowledge_graph(extraction_result['triples'])
    print(f"Update success: {update_success}")
    
    kg_stats = rc.knowledge_graph.get_stats()
    print(f"Knowledge graph stats: {kg_stats}\n")
    
    # Generate a response using the updated knowledge graph
    print("=== Step 3: Knowledge-Enhanced Response Generation ===")
    
    user_query = "What happened to Donald Trump at the rally in July 2024?"
    print(f"User query: {user_query}\n")
    
    print("=== Step 3.1: Knowledge-Free Response Generation ===")

    response = rc.generate_response(
        query=user_query,
        grounded=False,
    )
    
    log_result(response)
    
    print("Generated response:")
    print(response['text'])
    print()
        
    print("=== Step 3.2: Knowledge-Enhanced Response Generation ===")
    response = rc.generate_response(
        query=user_query,
        grounded=True,
        max_context_items=10,
    )
    print("Generated response:")
    print(response['text'])
    print()


    if response.get('meta', {}).get('has_contradictions', False):
        print("Contradictions detected:")
        for contradiction in response.get('meta', {}).get('contradictions', []):
            print(f"- Statement: {contradiction.get('statement')}")
            print(f"  Conflicts with: {contradiction.get('conflicting_fact')}")
        print()
        
        print("=== Step 4: Reflexive Correction ===")
        print("Attempting reflexive correction...")
        
        corrected_response = rc.target_llm.generate_with_reflexive_correction(
            query=user_query,
            context=rc.knowledge_graph.retrieve_context(user_query, max_items=10),
            validator=rc.validator,
            max_attempts=2
        )
        
        print("Corrected response:")
        print(corrected_response['text'])
        print()
    
    print("Example complete!")

if __name__ == "__main__":
    main()