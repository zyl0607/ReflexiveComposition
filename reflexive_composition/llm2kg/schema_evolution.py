# reflexive_composition/llm2kg/schema_evolution.py
"""
Schema evolution and management for the LLM2KG component.

Handles schema validation, suggestion, and evolution based on extracted knowledge.
"""

import logging
from typing import Dict, List, Any, Set, Optional, Tuple

logger = logging.getLogger(__name__)

class SchemaManager:
    """
    Manages knowledge graph schema evolution and validation.
    
    This class handles schema operations including:
    - Validating extracted knowledge against a schema
    - Suggesting schema updates based on extracted knowledge
    - Tracking schema evolution over time
    """
    
    def __init__(self):
        """Initialize the schema manager."""
        self.schema_history = []  # Track schema versions
    
    def validate_against_schema(self, 
                               extraction: Dict[str, Any], 
                               schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize extracted knowledge against a schema.
        
        Args:
            extraction: The extracted knowledge
            schema: The schema to validate against
            
        Returns:
            Validated and normalized extraction
        """
        # Get schema components
        entity_types = set(schema.get("entity_types", []))
        relationship_types = set(schema.get("relationship_types", []))
        
        # Track schema violations for reporting
        violations = []
        
        # Validate and normalize triples
        validated_triples = []
        unknown_entity_types = set()
        unknown_relationship_types = set()
        
        for triple in extraction.get("triples", []):
            # Check if triple structure is complete
            if not all(k in triple for k in ["subject", "predicate", "object"]):
                violations.append({
                    "triple": triple,
                    "reason": "Incomplete triple structure"
                })
                continue
            
            # Check relationship type
            relationship = triple["predicate"]
            if relationship_types and relationship not in relationship_types:
                unknown_relationship_types.add(relationship)
                triple["schema_violation"] = "unknown_relationship"
                violations.append({
                    "triple": triple,
                    "reason": f"Unknown relationship type: {relationship}"
                })
            
            # For entity types, we'd need additional information like
            # subject_type and object_type in the triple or schema
            if "subject_type" in triple and triple["subject_type"] not in entity_types:
                unknown_entity_types.add(triple["subject_type"])
                
            if "object_type" in triple and triple["object_type"] not in entity_types:
                unknown_entity_types.add(triple["object_type"])
            
            # Add to validated triples
            validated_triples.append(triple)
        
        # Return the validated extraction with violations
        return {
            "triples": validated_triples,
            "violations": violations,
            "unknown_entity_types": list(unknown_entity_types),
            "unknown_relationship_types": list(unknown_relationship_types)
        }
    
    def suggest_updates(self, 
                        extractions: List[Dict[str, Any]], 
                        current_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest schema updates based on extracted knowledge.
        
        Args:
            extractions: List of extraction results
            current_schema: Current schema definition
            
        Returns:
            Suggested schema updates
        """
        # Current schema components
        current_entity_types = set(current_schema.get("entity_types", []))
        current_relationship_types = set(current_schema.get("relationship_types", []))
        
        # Collect unknown types from all extractions
        all_unknown_entity_types = set()
        all_unknown_relationship_types = set()
        
        for extraction in extractions:
            all_unknown_entity_types.update(extraction.get("unknown_entity_types", []))
            all_unknown_relationship_types.update(extraction.get("unknown_relationship_types", []))
        
        # Generate suggestions
        suggestions = {
            "entity_types": [
                {
                    "type": entity_type,
                    "confidence": 0.7,  # Default confidence
                    "source_extractions": []  # Would include references to source extractions
                }
                for entity_type in all_unknown_entity_types
            ],
            "relationship_types": [
                {
                    "type": relationship_type,
                    "confidence": 0.7,  # Default confidence
                    "source_extractions": []  # Would include references to source extractions
                }
                for relationship_type in all_unknown_relationship_types
            ]
        }
        
        return suggestions
    
    def evolve_schema(self, 
                     current_schema: Dict[str, Any], 
                     approved_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update schema with approved changes.
        
        Args:
            current_schema: Current schema definition
            approved_updates: Approved schema updates
            
        Returns:
            Updated schema
        """
        # Create a new schema version
        new_schema = current_schema.copy()
        
        # Add entity types
        entity_types = set(new_schema.get("entity_types", []))
        for update in approved_updates.get("entity_types", []):
            entity_types.add(update["type"])
        new_schema["entity_types"] = sorted(list(entity_types))
        
        # Add relationship types
        relationship_types = set(new_schema.get("relationship_types", []))
        for update in approved_updates.get("relationship_types", []):
            relationship_types.add(update["type"])
        new_schema["relationship_types"] = sorted(list(relationship_types))
        
        # Add version history
        new_schema["version"] = current_schema.get("version", 0) + 1
        new_schema["updated_at"] = self._get_timestamp()
        
        # Store in history
        self.schema_history.append(new_schema)
        
        return new_schema
    
    def _get_timestamp(self) -> str:
        """
        Get current timestamp in ISO format.
        
        Returns:
            Current timestamp string
        """
        from datetime import datetime
        return datetime.utcnow().isoformat()
        
    def get_schema_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of schema versions.
        
        Returns:
            List of schema versions
        """
        return self.schema_history