# reflexive_composition/hitl/validator.py
"""
Validator component for human-in-the-loop oversight.

This module handles routing of validation tasks to human experts,
tracking of validation decisions, and integration with the knowledge graph.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union, Callable

logger = logging.getLogger(__name__)

class Validator:
    """
    Core validator for human-in-the-loop validation.
    
    This class manages the validation workflow, routing decisions to
    human experts and tracking their feedback.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 validation_interface: Optional[Any] = None):
        """
        Initialize the validator.
        
        Args:
            config: Configuration dictionary
            validation_interface: Interface for presenting validation tasks
        """
        self.config = config or {}
        
        # Set default confidence thresholds
        self.high_confidence_threshold = self.config.get("high_confidence_threshold", 0.9)
        self.low_confidence_threshold = self.config.get("low_confidence_threshold", 0.6)
        
        # Initialize validation interface
        if validation_interface:
            self.interface = validation_interface
        else:
            # Use default console interface if none provided
            from .interface import ConsoleValidationInterface
            self.interface = ConsoleValidationInterface()
        
        # Initialize validation router
        from .routing import ValidationRouter
        self.router = ValidationRouter(self.config)
        
        # Statistics
        self.stats = {
            "total_validations": 0,
            "accepted": 0,
            "rejected": 0,
            "modified": 0,
            "auto_accepted": 0,
            "auto_rejected": 0
        }
    
    def validate_triple(self, 
                        triple: Dict[str, Any],
                        source_text: str,
                        knowledge_graph: Any,
                        interactive: bool = True) -> Dict[str, Any]:
        """
        Validate a triple against source text and existing knowledge.
        
        Args:
            triple: Triple dictionary to validate
            source_text: Source text the triple was extracted from
            knowledge_graph: Knowledge graph instance
            
        Returns:
            Validation result
        """
        # Check confidence for auto-handling
        confidence = triple.get("confidence", 0.0)
        
        # Auto-accept high-confidence triples
        if confidence >= self.high_confidence_threshold:
            self.stats["auto_accepted"] += 1
            return {
                "accepted": True,
                "triple": triple,
                "validation_type": "auto_accept",
                "reason": f"Confidence {confidence} above threshold {self.high_confidence_threshold}"
            }
        
        # Auto-reject very low-confidence triples
        if confidence < self.low_confidence_threshold:
            self.stats["auto_rejected"] += 1
            return {
                "accepted": False,
                "triple": triple,
                "validation_type": "auto_reject",
                "reason": f"Confidence {confidence} below threshold {self.low_confidence_threshold}"
            }
        
        # Check for contradiction with existing knowledge
        contradiction = self._check_contradiction(triple, knowledge_graph)
        
        # Prepare validation context
        context = {
            "triple": triple,
            "source_text": source_text,
            "contradiction": contradiction,
            "knowledge_graph": knowledge_graph
        }
        
        # Route to appropriate validator based on type
        validator_type = self._determine_validator_type(triple, contradiction)
        
        # Present to human validator
        validation_result = self.router.route_validation(
            validator_type, context, self.interface, interactive=interactive
        )
        
        # Update statistics
        self.stats["total_validations"] += 1
        if validation_result.get("accepted", False):
            self.stats["accepted"] += 1
        else:
            self.stats["rejected"] += 1
        
        if validation_result.get("modified", False):
            self.stats["modified"] += 1
        
        return validation_result
    
    def validate_triples(self, 
                       triples: List[Dict[str, Any]], 
                       source_text: str, 
                       knowledge_graph: Any,
                       interactive: bool = True) -> Dict[str, Any]:
        """
        Validate multiple triples.
        
        Args:
            triples: List of triple dictionaries to validate
            source_text: Source text the triples were extracted from
            knowledge_graph: Knowledge graph instance
            
        Returns:
            Validation results
        """
        results = []
        
        for triple in triples:
            result = self.validate_triple(triple, source_text, knowledge_graph)
            results.append(result)
        
        return {
            "results": results,
            "accepted_count": sum(1 for r in results if r.get("accepted", False)),
            "rejected_count": sum(1 for r in results if not r.get("accepted", False)),
            "modified_count": sum(1 for r in results if r.get("modified", False))
        }
    
    def validate_schema_update(self, 
                              schema_update: Dict[str, Any], 
                              current_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a proposed schema update.
        
        Args:
            schema_update: Proposed schema update
            current_schema: Current schema definition
            
        Returns:
            Validation result
        """
        # Prepare validation context
        context = {
            "schema_update": schema_update,
            "current_schema": current_schema
        }
        
        # Present to human validator
        validation_result = self.router.route_validation(
            "schema", context, self.interface
        )
        
        # Update statistics
        self.stats["total_validations"] += 1
        if validation_result.get("accepted", False):
            self.stats["accepted"] += 1
        else:
            self.stats["rejected"] += 1
        
        return validation_result
    
    def validate_response(self, 
                         response: Dict[str, Any], 
                         contradictions: List[Dict[str, Any]], 
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate an LLM response against KG context.
        
        Args:
            response: Generated response
            contradictions: Detected contradictions
            context: KG context used for generation
            
        Returns:
            Validation result
        """
        # If no contradictions, nothing to validate
        if not contradictions:
            return response
        
        # Prepare validation context
        validation_context = {
            "response": response,
            "contradictions": contradictions,
            "kg_context": context
        }
        
        # Present to human validator
        validation_result = self.router.route_validation(
            "response", validation_context, self.interface
        )
        
        # Update statistics
        self.stats["total_validations"] += 1
        if validation_result.get("accepted", False):
            self.stats["accepted"] += 1
        else:
            self.stats["rejected"] += 1
        
        if validation_result.get("modified", False):
            self.stats["modified"] += 1
            response["text"] = validation_result.get("corrected_text", response.get("text", ""))
            response["validated"] = True
            response["validator_notes"] = validation_result.get("notes", "")
        
        return response
    
    def validate_with_feedback(self, 
                              triples: List[Dict[str, Any]], 
                              feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate triples using provided feedback.
        
        Args:
            triples: List of triple dictionaries
            feedback: Feedback dictionary
            
        Returns:
            Validation results
        """
        feedback_type = feedback.get("type", "explicit")
        
        if feedback_type == "explicit":
            # Explicit feedback provides direct accept/reject decisions
            return self._validate_with_explicit_feedback(triples, feedback)
        elif feedback_type == "implicit":
            # Implicit feedback requires interpretation
            return self._validate_with_implicit_feedback(triples, feedback)
        else:
            logger.warning(f"Unsupported feedback type: {feedback_type}")
            return {"triples": [], "validation_type": "failed"}
    
    def _validate_with_explicit_feedback(self, 
                                       triples: List[Dict[str, Any]], 
                                       feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate triples using explicit feedback.
        
        Args:
            triples: List of triple dictionaries
            feedback: Explicit feedback dictionary
            
        Returns:
            Validation results
        """
        accepted_triples = []
        feedback_decisions = feedback.get("decisions", {})
        
        for triple in triples:
            # Generate a triple key for lookup
            triple_key = json.dumps({
                "subject": triple.get("subject", ""),
                "predicate": triple.get("predicate", ""),
                "object": triple.get("object", "")
            })
            
            # Check if the triple has an explicit decision
            if triple_key in feedback_decisions:
                decision = feedback_decisions[triple_key]
                if decision.get("accepted", False):
                    # Use modified triple if available
                    if "modified_triple" in decision:
                        accepted_triples.append(decision["modified_triple"])
                    else:
                        accepted_triples.append(triple)
            elif triple.get("confidence", 0.0) >= self.high_confidence_threshold:
                # Auto-accept high-confidence triples without explicit feedback
                accepted_triples.append(triple)
        
        return {
            "triples": accepted_triples,
            "validation_type": "explicit_feedback",
            "validated_count": len(accepted_triples)
        }
    
    def _validate_with_implicit_feedback(self, 
                                       triples: List[Dict[str, Any]], 
                                       feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate triples using implicit feedback.
        
        Args:
            triples: List of triple dictionaries
            feedback: Implicit feedback dictionary
            
        Returns:
            Validation results
        """
        # For implicit feedback, we need to interpret signals
        # This is a simplified version
        accepted_triples = []
        
        # Extract feedback signals
        relevance_rating = feedback.get("relevance", 0)
        usefulness_rating = feedback.get("usefulness", 0)
        
        # Derive a threshold based on feedback signals
        threshold_adjustment = (relevance_rating + usefulness_rating) / 20.0  # Scale to -0.5 to 0.5
        adjusted_threshold = self.high_confidence_threshold - threshold_adjustment
        
        # Apply adjusted threshold
        for triple in triples:
            if triple.get("confidence", 0.0) >= adjusted_threshold:
                accepted_triples.append(triple)
        
        return {
            "triples": accepted_triples,
            "validation_type": "implicit_feedback",
            "validated_count": len(accepted_triples),
            "adjusted_threshold": adjusted_threshold
        }
    
    def _check_contradiction(self, 
                            triple: Dict[str, Any], 
                            knowledge_graph: Any) -> Optional[Dict[str, Any]]:
        """
        Check if a triple contradicts existing knowledge.
        
        Args:
            triple: Triple dictionary
            knowledge_graph: Knowledge graph instance
            
        Returns:
            Contradiction info if found, None otherwise
        """
        # Extract triple components
        subject = triple.get("subject")
        predicate = triple.get("predicate")
        obj = triple.get("object")
        
        if not all([subject, predicate, obj]):
            return None
        
        # Query for existing triples with same subject and predicate
        query = f"{subject}:{predicate}"
        existing_triples = knowledge_graph.query(query, "entity_predicate")
        
        # Check for contradictions
        for existing in existing_triples:
            existing_obj = existing.get("object")
            
            # Skip if objects match
            if existing_obj == obj:
                continue
            
            # For simple literals, direct comparison
            if isinstance(obj, (str, int, float)) and isinstance(existing_obj, (str, int, float)):
                return {
                    "existing_triple": existing,
                    "new_triple": triple,
                    "reason": "Conflicting object values"
                }
        
        return None
    
    def _determine_validator_type(self, 
                                triple: Dict[str, Any], 
                                contradiction: Optional[Dict[str, Any]]) -> str:
        """
        Determine the appropriate validator type for a triple.
        
        Args:
            triple: Triple dictionary
            contradiction: Contradiction info if found
            
        Returns:
            Validator type
        """
        if contradiction:
            return "contradiction"
        
        # Check for schema violations
        if "schema_violation" in triple:
            return "schema"
        
        # Default to general validation
        return "general"
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get validation statistics.
        
        Returns:
            Validation statistics
        """
        return self.stats
