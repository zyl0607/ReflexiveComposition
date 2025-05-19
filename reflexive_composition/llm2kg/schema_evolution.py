# reflexive_composition/llm2kg/schema_evolution.py
"""
Schema Evolution Framework for dynamic knowledge graph schema management.

This module handles the automated suggestion and validation of schema updates
based on extracted knowledge and domain requirements.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SchemaUpdate:
    """
    Represents a proposed schema update with provenance and confidence.
    """
    entity_types: List[Dict[str, Any]]
    relationship_types: List[Dict[str, Any]]
    attribute_modifications: List[Dict[str, Any]]
    source_extractions: List[Dict[str, Any]]
    confidence: float
    rationale: str

class SchemaManager:
    """
    Manages the evolution of knowledge graph schemas through suggested updates,
    validation workflows, and controlled implementation.
    """
    
    def __init__(self, 
                 initial_schema: Optional[Dict[str, Any]] = None,
                 kb_llm: Optional[Any] = None):
        """
        Initialize the schema manager.
        
        Args:
            initial_schema: Initial schema definition
            kb_llm: Knowledge Builder LLM instance for schema suggestion
        """
        self.current_schema = initial_schema or {
            "entity_types": [],
            "relationship_types": [],
            "version": 0
        }
        self.kb_llm = kb_llm
        self.schema_history = [self.current_schema.copy()]
        
    def suggest_updates(self, 
                       extractions: List[Dict[str, Any]],
                       current_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest updates to the schema based on extracted knowledge.
        
        Args:
            extractions: List of extracted knowledge
            current_schema: Current schema definition
            
        Returns:
            Suggested schema updates
        """
        # Analyze extractions for new entity types
        entity_type_candidates = self._extract_entity_types(extractions, current_schema)
        
        # Analyze extractions for new relationship types
        relationship_type_candidates = self._extract_relationship_types(extractions, current_schema)
        
        # Analyze attribute consistency and suggest modifications
        attribute_candidates = self._analyze_attributes(extractions, current_schema)
        
        # Compile suggestions with confidence scores
        updates = {
            "entity_types": entity_type_candidates,
            "relationship_types": relationship_type_candidates,
            "attribute_modifications": attribute_candidates,
            "source_extractions": extractions[:5],  # Add sample evidence
            "version": current_schema.get("version", 0) + 1
        }
        
        return updates
    
    def _extract_entity_types(self, 
                             extractions: List[Dict[str, Any]], 
                             current_schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract potential new entity types from knowledge extractions.
        
        Args:
            extractions: List of extracted knowledge
            current_schema: Current schema definition
            
        Returns:
            Potential new entity types with confidence scores
        """
        # First, get existing entity types
        existing_types = set(current_schema.get("entity_types", []))
        
        # Collect all entity types mentioned in extractions
        candidate_types = {}
        
        for extraction in extractions:
            # Handle triples case
            if "triples" in extraction:
                for triple in extraction["triples"]:
                    # Look for explicit type statements
                    if triple.get("predicate", "").lower() in ["type", "is_a", "instance_of", "rdf:type"]:
                        entity_type = triple.get("object", "")
                        if entity_type and entity_type not in existing_types:
                            if entity_type not in candidate_types:
                                candidate_types[entity_type] = {
                                    "type": entity_type,
                                    "confidence": float(triple.get("confidence", 0.5)),
                                    "count": 1,
                                    "source_extractions": [triple]
                                }
                            else:
                                candidate_types[entity_type]["count"] += 1
                                candidate_types[entity_type]["confidence"] = max(
                                    candidate_types[entity_type]["confidence"],
                                    float(triple.get("confidence", 0.5))
                                )
                                candidate_types[entity_type]["source_extractions"].append(triple)
        
        # Filter candidates by frequency and confidence
        return [
            {
                "type": cand["type"],
                "confidence": cand["confidence"],
                "frequency": cand["count"],
                "source_extractions": cand["source_extractions"][:3]  # Limit evidence to avoid bloat
            }
            for cand in candidate_types.values()
            if cand["count"] >= 2 and cand["confidence"] >= 0.6  # Minimum thresholds
        ]
    
    def _extract_relationship_types(self, 
                                  extractions: List[Dict[str, Any]], 
                                  current_schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract potential new relationship types from knowledge extractions.
        
        Args:
            extractions: List of extracted knowledge
            current_schema: Current schema definition
            
        Returns:
            Potential new relationship types with confidence scores
        """
        # Similar to entity types but focusing on predicates
        existing_relationships = set(current_schema.get("relationship_types", []))
        
        # Collect all relationship types mentioned in extractions
        candidate_relationships = {}
        
        for extraction in extractions:
            if "triples" in extraction:
                for triple in extraction["triples"]:
                    relationship = triple.get("predicate", "")
                    if relationship and relationship not in existing_relationships:
                        if relationship not in candidate_relationships:
                            candidate_relationships[relationship] = {
                                "type": relationship,
                                "confidence": float(triple.get("confidence", 0.5)),
                                "count": 1,
                                "source_extractions": [triple],
                                "subject_types": set([triple.get("subject_type", "unknown")]),
                                "object_types": set([triple.get("object_type", "unknown")])
                            }
                        else:
                            candidate_relationships[relationship]["count"] += 1
                            candidate_relationships[relationship]["confidence"] = max(
                                candidate_relationships[relationship]["confidence"],
                                float(triple.get("confidence", 0.5))
                            )
                            candidate_relationships[relationship]["source_extractions"].append(triple)
                            if "subject_type" in triple:
                                candidate_relationships[relationship]["subject_types"].add(triple["subject_type"])
                            if "object_type" in triple:
                                candidate_relationships[relationship]["object_types"].add(triple["object_type"])
        
        # Convert to list format with domain/range info
        return [
            {
                "type": rel["type"],
                "confidence": rel["confidence"],
                "frequency": rel["count"],
                "domain": list(rel["subject_types"]) if len(rel["subject_types"]) > 0 and "unknown" not in rel["subject_types"] else [],
                "range": list(rel["object_types"]) if len(rel["object_types"]) > 0 and "unknown" not in rel["object_types"] else [],
                "source_extractions": rel["source_extractions"][:3]
            }
            for rel in candidate_relationships.values()
            if rel["count"] >= 2 and rel["confidence"] >= 0.6
        ]
    
    def _analyze_attributes(self, 
                          extractions: List[Dict[str, Any]], 
                          current_schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze attribute patterns and suggest modifications.
        
        Args:
            extractions: List of extracted knowledge
            current_schema: Current schema definition
            
        Returns:
            Suggested attribute modifications
        """
        # This would analyze attribute patterns and suggest modifications
        # Simplified implementation for now
        return []
    
    def validate_against_schema(self, 
                               extraction: Dict[str, Any], 
                               schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted knowledge against the schema.
        
        Args:
            extraction: Extracted knowledge
            schema: Schema definition
            
        Returns:
            Validated knowledge with schema violations flagged
        """
        # Create a copy to avoid modifying the original
        validated = {"triples": []}
        
        # Get schema constraints
        entity_types = set(schema.get("entity_types", []))
        relationship_types = set(schema.get("relationship_types", []))
        
        for triple in extraction.get("triples", []):
            # Create a copy of the triple
            validated_triple = triple.copy()
            
            # Check for schema violations
            if "subject_type" in triple and triple["subject_type"] not in entity_types:
                validated_triple["schema_violation"] = {
                    "type": "unknown_entity_type",
                    "entity_type": triple["subject_type"],
                    "position": "subject"
                }
            
            if "object_type" in triple and triple["object_type"] not in entity_types and triple["predicate"].lower() not in ["type", "is_a"]:
                validated_triple["schema_violation"] = {
                    "type": "unknown_entity_type",
                    "entity_type": triple["object_type"],
                    "position": "object"
                }
            
            if triple["predicate"] not in relationship_types:
                validated_triple["schema_violation"] = {
                    "type": "unknown_relationship_type",
                    "relationship_type": triple["predicate"]
                }
            
            validated["triples"].append(validated_triple)
        
        return validated
    
    def apply_update(self, 
                    update: Dict[str, Any], 
                    current_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply validated updates to the schema.
        
        Args:
            update: Validated schema update
            current_schema: Current schema definition
            
        Returns:
            Updated schema
        """
        # Create a copy of the current schema
        updated_schema = current_schema.copy()
        
        # Update entity types
        existing_entity_types = set(current_schema.get("entity_types", []))
        for entity_type in update.get("entity_types", []):
            if entity_type["type"] not in existing_entity_types:
                if "entity_types" not in updated_schema:
                    updated_schema["entity_types"] = []
                updated_schema["entity_types"].append(entity_type["type"])
        
        # Update relationship types
        existing_relationship_types = set(current_schema.get("relationship_types", []))
        for relationship_type in update.get("relationship_types", []):
            if relationship_type["type"] not in existing_relationship_types:
                if "relationship_types" not in updated_schema:
                    updated_schema["relationship_types"] = []
                updated_schema["relationship_types"].append(relationship_type["type"])
        
        # Update version
        updated_schema["version"] = current_schema.get("version", 0) + 1
        
        # Update timestamp
        from datetime import datetime
        updated_schema["updated_at"] = datetime.utcnow().isoformat()
        
        # Store in history
        self.schema_history.append(updated_schema.copy())
        self.current_schema = updated_schema
        
        return updated_schema
    
    def generate_schema_prompt(self, 
                              domain_description: str, 
                              existing_triples: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate a prompt for schema generation using LLM assistance.
        
        Args:
            domain_description: Description of the domain for schema generation
            existing_triples: Optional list of existing triples to inform schema
            
        Returns:
            Prompt for schema generation
        """
        triple_examples = ""
        if existing_triples:
            # Format some example triples to include in the prompt
            triple_examples = "Example triples from the domain:\n"
            for i, triple in enumerate(existing_triples[:5], 1):
                triple_examples += f"{i}. {triple.get('subject', '')} - {triple.get('predicate', '')} - {triple.get('object', '')}\n"
        
        # Create the prompt for schema generation
        prompt = f"""Generate a knowledge graph schema for the following domain:

{domain_description}

{triple_examples}

The schema should include:
1. Entity types - Classes of objects in the domain
2. Relationship types - Types of relationships between entities
3. Key attributes - Important properties for each entity type

Format the output as JSON with the following structure:
{{
  "entity_types": ["Type1", "Type2", ...],
  "relationship_types": ["Relation1", "Relation2", ...],
  "attributes": [
    {{"entity_type": "Type1", "attributes": ["attr1", "attr2", ...]}},
    ...
  ]
}}

Focus on the core concepts and relationships that would be most useful for organizing knowledge in this domain."""
        
        return prompt
    
    def suggest_schema_from_description(self, 
                                       domain_description: str,
                                       kb_llm: Any) -> Dict[str, Any]:
        """
        Generate a schema suggestion using LLM from domain description.
        
        Args:
            domain_description: Description of the domain
            kb_llm: Knowledge Builder LLM instance
            
        Returns:
            Suggested schema
        """
        if not kb_llm:
            if not self.kb_llm:
                raise ValueError("KB LLM not provided for schema generation")
            kb_llm = self.kb_llm
            
        # Generate prompt for schema suggestion
        prompt = self.generate_schema_prompt(domain_description)
        
        # Get LLM to generate schema suggestion
        schema_suggestion = kb_llm.extract(prompt, None)
        
        # Process the result
        if isinstance(schema_suggestion, dict):
            # Add version and timestamp
            schema_suggestion["version"] = 1
            from datetime import datetime
            schema_suggestion["created_at"] = datetime.utcnow().isoformat()
            
            return schema_suggestion
        else:
            # Handle error case
            logger.error(f"Failed to generate schema suggestion: {schema_suggestion}")
            return {
                "entity_types": [],
                "relationship_types": [],
                "version": 1
            }
