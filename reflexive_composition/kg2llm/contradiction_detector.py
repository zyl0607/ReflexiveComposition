# reflexive_composition/kg2llm/contradiction_detector.py
"""
Contradiction detection for knowledge graph enhanced LLM inference.

This module handles the detection of contradictions between LLM-generated
responses and knowledge graph context.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set, Tuple

logger = logging.getLogger(__name__)

class ContradictionDetector:
    """
    Detects contradictions between generated text and knowledge context.
    
    This class implements various strategies for identifying contradictions
    and inconsistencies between LLM-generated text and the knowledge graph
    context it was conditioned on.
    """
    
    def __init__(self):
        """Initialize the contradiction detector."""
        pass
    
    def detect_contradictions(self, 
                            generated_text: str, 
                            context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect contradictions between generated text and knowledge context.
        
        Args:
            generated_text: LLM-generated text
            context: Knowledge graph context used for generation
            
        Returns:
            List of detected contradictions
        """
        contradictions = []
        
        # Extract statements from generated text
        statements = self._extract_statements(generated_text)
        
        # Check different types of context
        if "triples" in context:
            # Check contradictions against triples
            triple_contradictions = self._check_triple_contradictions(
                statements, context["triples"]
            )
            contradictions.extend(triple_contradictions)
        
        if "entities" in context:
            # Check contradictions against entities
            entity_contradictions = self._check_entity_contradictions(
                statements, context["entities"]
            )
            contradictions.extend(entity_contradictions)
        
        return contradictions
    
    def _extract_statements(self, text: str) -> List[str]:
        """
        Extract individual statements from generated text.
        
        Args:
            text: Generated text
            
        Returns:
            List of extracted statements
        """
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Remove empty sentences
        statements = [s.strip() for s in sentences if s.strip()]
        
        return statements
    
    def _check_triple_contradictions(self, 
                                   statements: List[str], 
                                   triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Check for contradictions against knowledge graph triples.
        
        Args:
            statements: Extracted statements from generated text
            triples: Knowledge graph triples
            
        Returns:
            List of detected contradictions
        """
        contradictions = []
        
        # Extract subjects from triples for faster lookup
        triple_subjects = {}
        triple_predicates = {}
        
        for triple in triples:
            subject = triple.get("subject", "").lower()
            predicate = triple.get("predicate", "").lower()
            obj = triple.get("object", "").lower()
            
            # Index by subject
            if subject not in triple_subjects:
                triple_subjects[subject] = []
            triple_subjects[subject].append((predicate, obj, triple))
            
            # Index by predicate
            if predicate not in triple_predicates:
                triple_predicates[predicate] = []
            triple_predicates[predicate].append((subject, obj, triple))
        
        # Check each statement for potential contradictions
        for statement in statements:
            # Lowercase for case-insensitive matching
            statement_lower = statement.lower()
            
            # Check for matches with triple subjects
            for subject, subject_triples in triple_subjects.items():
                if subject in statement_lower:
                    # Subject match found, check for predicate and object
                    for predicate, obj, triple in subject_triples:
                        if predicate in statement_lower:
                            # Negative check - look for statements that negate the triple
                            # For example, "X is not Y" contradicts "X is Y"
                            negation_patterns = [
                                f"{subject} is not {obj}",
                                f"{subject} isn't {obj}",
                                f"{subject} doesn't {predicate}",
                                f"{subject} does not {predicate}",
                                f"{subject} never {predicate}",
                                f"not {predicate} {obj}"
                            ]
                            
                            for pattern in negation_patterns:
                                if pattern in statement_lower:
                                    # Found a contradiction
                                    contradictions.append({
                                        "statement": statement,
                                        "conflicting_fact": f"{subject} {predicate} {obj}",
                                        "triple": triple,
                                        "contradiction_type": "negation"
                                    })
                            
                            # Alternative value check
                            # For example, "X is Z" contradicts "X is Y" if Z ≠ Y
                            match = re.search(f"{subject} {predicate} ([^.!?]+)", statement_lower)
                            if match:
                                stated_obj = match.group(1).strip()
                                
                                # Check if the stated object differs from the known object
                                if stated_obj != obj and not self._is_compatible(stated_obj, obj):
                                    contradictions.append({
                                        "statement": statement,
                                        "conflicting_fact": f"{subject} {predicate} {obj}",
                                        "stated_value": stated_obj,
                                        "triple": triple,
                                        "contradiction_type": "alternative_value"
                                    })
        
        return contradictions
    
    def _check_entity_contradictions(self, 
                                  statements: List[str], 
                                  entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Check for contradictions against entity information.
        
        Args:
            statements: Extracted statements from generated text
            entities: Entity information
            
        Returns:
            List of detected contradictions
        """
        contradictions = []
        
        # Index entities by ID for faster lookup
        entity_map = {entity.get("id", "").lower(): entity for entity in entities}
        
        # Check each statement for potential contradictions
        for statement in statements:
            # Lowercase for case-insensitive matching
            statement_lower = statement.lower()
            
            # Check for matches with entity IDs
            for entity_id, entity in entity_map.items():
                if entity_id in statement_lower:
                    # Entity match found, check properties
                    properties = entity.get("properties", {})
                    
                    for prop_name, prop_value in properties.items():
                        prop_name_lower = prop_name.lower()
                        prop_value_lower = str(prop_value).lower()
                        
                        if prop_name_lower in statement_lower:
                            # Negative check
                            negation_patterns = [
                                f"{entity_id} does not have {prop_name_lower}",
                                f"{entity_id} doesn't have {prop_name_lower}",
                                f"{entity_id} has no {prop_name_lower}",
                                f"{prop_name_lower} of {entity_id} is not {prop_value_lower}",
                                f"{prop_name_lower} of {entity_id} isn't {prop_value_lower}"
                            ]
                            
                            for pattern in negation_patterns:
                                if pattern in statement_lower:
                                    # Found a contradiction
                                    contradictions.append({
                                        "statement": statement,
                                        "conflicting_fact": f"{entity_id} has {prop_name} = {prop_value}",
                                        "entity": entity,
                                        "property": prop_name,
                                        "contradiction_type": "property_negation"
                                    })
                            
                            # Alternative value check
                            match = re.search(f"{prop_name_lower} of {entity_id} is ([^.!?]+)", statement_lower)
                            if not match:
                                match = re.search(f"{entity_id} has {prop_name_lower} ([^.!?]+)", statement_lower)
                                
                            if match:
                                stated_value = match.group(1).strip()
                                
                                # Check if the stated value differs from the known value
                                if stated_value != prop_value_lower and not self._is_compatible(stated_value, prop_value_lower):
                                    contradictions.append({
                                        "statement": statement,
                                        "conflicting_fact": f"{entity_id} has {prop_name} = {prop_value}",
                                        "stated_value": stated_value,
                                        "entity": entity,
                                        "property": prop_name,
                                        "contradiction_type": "property_alternative_value"
                                    })
        
        return contradictions
    
    def _is_compatible(self, value1: str, value2: str) -> bool:
        """
        Check if two values are semantically compatible.
        
        Args:
            value1: First value
            value2: Second value
            
        Returns:
            True if values are compatible, False otherwise
        """
        # Normalize for comparison
        v1 = value1.strip().lower()
        v2 = value2.strip().lower()
        
        # Check for exact match
        if v1 == v2:
            return True
        
        # Check for numeric equivalence
        if self._is_numeric(v1) and self._is_numeric(v2):
            try:
                n1 = float(v1)
                n2 = float(v2)
                # Allow small differences for numeric values
                return abs(n1 - n2) < 0.001
            except ValueError:
                pass
        
        # Check for substring relationship
        if v1 in v2 or v2 in v1:
            return True
        
        # Check for common aliases and abbreviations
        # This is a simplified example - would need to be expanded
        # based on domain-specific knowledge
        aliases = {
            "united states": ["us", "usa", "u.s.", "u.s.a."],
            "united kingdom": ["uk", "u.k.", "britain", "great britain"],
            # Add more aliases as needed
        }
        
        # Check if the values are aliases of each other
        for term, alias_list in aliases.items():
            if v1 == term and v2 in alias_list:
                return True
            if v2 == term and v1 in alias_list:
                return True
            if v1 in alias_list and v2 in alias_list:
                return True
        
        # Default to not compatible
        return False
    
    def _is_numeric(self, value: str) -> bool:
        """
        Check if a value is numeric.
        
        Args:
            value: Value to check
            
        Returns:
            True if value is numeric, False otherwise
        """
        try:
            float(value)
            return True
        except ValueError:
            return False