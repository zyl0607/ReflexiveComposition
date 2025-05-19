# reflexive_composition/knowledge_graph/update_triggers.py
"""
Update trigger mechanisms for knowledge graph maintenance.

This module provides mechanisms for detecting when knowledge graph updates
are needed based on queries, feedback, and validation signals.
"""

import logging
import re
import json
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class TriggerType(Enum):
    """Types of update triggers."""
    QUERY_FAILURE = "query_failure"
    SOURCE_UPDATE = "source_update"
    SCHEMA_EVOLUTION = "schema_evolution"
    VALIDATION_FEEDBACK = "validation_feedback"
    CONTRADICTION = "contradiction"


@dataclass
class UpdateTrigger:
    """Represents a knowledge graph update trigger."""
    type: TriggerType
    source: str
    confidence: float
    details: Dict[str, Any]
    timestamp: str


class QueryBasedTrigger:
    """
    Detects and processes query-based update triggers.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 kb_llm = None):
        """
        Initialize the query-based trigger detector.
        
        Args:
            confidence_threshold: Confidence threshold for triggers
            kb_llm: Knowledge Builder LLM for extraction
        """
        self.confidence_threshold = confidence_threshold
        self.kb_llm = kb_llm
        
        # Track query failures
        self.query_failure_stats = {
            "total_queries": 0,
            "failed_queries": 0,
            "triggered_updates": 0
        }
    
    def detect_query_failure(self, 
                            query: str, 
                            response: Dict[str, Any]) -> Optional[UpdateTrigger]:
        """
        Detect if a query response indicates a knowledge gap.
        
        Args:
            query: User query
            response: Generated response
            
        Returns:
            Update trigger if failure detected, None otherwise
        """
        self.query_failure_stats["total_queries"] += 1
        
        # Check for explicit indicators of knowledge gaps
        failure_indicators = [
            "I don't have information",
            "I don't know",
            "I don't have data",
            "I don't have knowledge",
            "I don't have access",
            "I'm not aware",
            "no information available",
            "couldn't find information",
            "no data available",
            "no knowledge about",
            "beyond my knowledge",
            "no context provided"
        ]
        
        response_text = response.get("text", "")
        meta = response.get("meta", {})
        
        # Check for contradictions
        has_contradictions = meta.get("has_contradictions", False)
        contradiction_details = meta.get("contradictions", [])
        
        # Check for explicit failure indicators
        explicit_failure = any(indicator.lower() in response_text.lower() for indicator in failure_indicators)
        
        # Check for low confidence
        low_confidence = meta.get("confidence", 1.0) < 0.5
        
        # If contradiction detected
        if has_contradictions:
            self.query_failure_stats["failed_queries"] += 1
            trigger = UpdateTrigger(
                type=TriggerType.CONTRADICTION,
                source="query_response",
                confidence=0.9,
                details={
                    "query": query,
                    "response_summary": response_text[:100] + "..." if len(response_text) > 100 else response_text,
                    "contradictions": contradiction_details
                },
                timestamp=self._get_timestamp()
            )
            self.query_failure_stats["triggered_updates"] += 1
            return trigger
        
        # If explicit failure detected
        if explicit_failure:
            self.query_failure_stats["failed_queries"] += 1
            trigger = UpdateTrigger(
                type=TriggerType.QUERY_FAILURE,
                source="explicit_gap",
                confidence=0.8,
                details={
                    "query": query,
                    "response_summary": response_text[:100] + "..." if len(response_text) > 100 else response_text,
                    "indicator": next((i for i in failure_indicators if i.lower() in response_text.lower()), "")
                },
                timestamp=self._get_timestamp()
            )
            self.query_failure_stats["triggered_updates"] += 1
            return trigger
        
        # If low confidence detected
        if low_confidence:
            self.query_failure_stats["failed_queries"] += 1
            trigger = UpdateTrigger(
                type=TriggerType.QUERY_FAILURE,
                source="low_confidence",
                confidence=0.7,
                details={
                    "query": query,
                    "response_summary": response_text[:100] + "..." if len(response_text) > 100 else response_text,
                    "model_confidence": meta.get("confidence", 0.0)
                },
                timestamp=self._get_timestamp()
            )
            self.query_failure_stats["triggered_updates"] += 1
            return trigger
        
        # No failure detected
        return None
    
    def process_trigger(self, 
                       trigger: UpdateTrigger, 
                       knowledge_sources: List[str]) -> Dict[str, Any]:
        """
        Process an update trigger to extract new knowledge.
        
        Args:
            trigger: Update trigger
            knowledge_sources: List of source text to extract from
            
        Returns:
            Extracted knowledge
        """
        if not self.kb_llm:
            logger.error("KB LLM not available for knowledge extraction")
            return {"triples": []}
        
        # Process based on trigger type
        if trigger.type == TriggerType.QUERY_FAILURE:
            # Extract focused knowledge based on query
            query = trigger.details.get("query", "")
            
            # Generate extraction prompt focusing on the query topic
            extraction_results = []
            
            for source in knowledge_sources:
                # Generate an extraction prompt focused on the query
                from reflexive_composition.llm2kg.prompt_templates import ExtractionPrompts
                
                # Determine the type of extraction needed
                if self._is_temporal_query(query):
                    prompt = ExtractionPrompts.temporal_extraction(source)
                else:
                    prompt = ExtractionPrompts.general_extraction(source)
                
                # Extract knowledge
                result = self.kb_llm.extract(prompt)
                
                if result and "triples" in result and result["triples"]:
                    extraction_results.extend(result["triples"])
            
            # Return combined results
            return {"triples": extraction_results}
            
        elif trigger.type == TriggerType.CONTRADICTION:
            # Focus extraction on resolving the contradiction
            contradictions = trigger.details.get("contradictions", [])
            
            if not contradictions:
                return {"triples": []}
            
            # Extract knowledge focusing on contradictory facts
            extraction_results = []
            
            # Process each contradiction
            for contradiction in contradictions:
                statement = contradiction.get("statement", "")
                conflicting_fact = contradiction.get("conflicting_fact", "")
                
                # Generate targeted extraction for each source
                for source in knowledge_sources:
                    # Build a focused prompt addressing the contradiction
                    prompt = f"""
Extract knowledge from the following text to resolve a potential contradiction.

Contradiction:
- Statement: {statement}
- Conflicting fact: {conflicting_fact}

Source text:
{source}

Focus on extracting precise information about this specific contradiction.
Return the extracted information as JSON with entities, relationships, and attributes.

Expected output format:
{{
    "triples": [
        {{
            "subject": "entity_name",
            "predicate": "relation_type",
            "object": "target_entity_or_value",
            "confidence": 0.95
        }}
    ]
}}
"""
                    # Extract knowledge
                    result = self.kb_llm.extract(prompt)
                    
                    if result and "triples" in result and result["triples"]:
                        # Mark these triples as contradiction resolutions
                        for triple in result["triples"]:
                            triple["source"] = "contradiction_resolution"
                            triple["contradicts"] = conflicting_fact
                        
                        extraction_results.extend(result["triples"])
            
            # Return combined results
            return {"triples": extraction_results}
        
        # Default empty response for unsupported trigger types
        return {"triples": []}
    
    def _is_temporal_query(self, query: str) -> bool:
        """
        Determine if a query is related to temporal information.
        
        Args:
            query: User query
            
        Returns:
            True if temporal query, False otherwise
        """
        temporal_indicators = [
            r'\bwhen\b',
            r'\bdate\b',
            r'\btime\b',
            r'\byear\b',
            r'\bmonth\b',
            r'\bday\b',
            r'\bearlier\b',
            r'\blater\b',
            r'\bbefore\b',
            r'\bafter\b',
            r'\bduring\b',
            r'\brecently\b',
            r'\blatest\b',
            r'\bcurrent\b',
            r'\bpast\b',
            r'\bfuture\b',
            r'\bhistory\b',
            r'\btimeline\b'
        ]
        
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in temporal_indicators)
    
    def _get_timestamp(self) -> str:
        """
        Get current timestamp in ISO format.
        
        Returns:
            Current timestamp string
        """
        return datetime.utcnow().isoformat()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about query-based triggers.
        
        Returns:
            Trigger statistics
        """
        return self.query_failure_stats


class ValidationFeedbackTrigger:
    """
    Detects and processes validation feedback-based update triggers.
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the validation feedback trigger detector.
        
        Args:
            confidence_threshold: Confidence threshold for triggers
        """
        self.confidence_threshold = confidence_threshold
        
        # Track validation stats
        self.validation_stats = {
            "total_validations": 0,
            "rejected": 0,
            "modified": 0,
            "triggered_updates": 0
        }
    
    def detect_validation_trigger(self, 
                                validation_result: Dict[str, Any]) -> Optional[UpdateTrigger]:
        """
        Detect if validation feedback indicates a knowledge update need.
        
        Args:
            validation_result: Result from validator
            
        Returns:
            Update trigger if detected, None otherwise
        """
        self.validation_stats["total_validations"] += 1
        
        # Check if validation was rejected
        if not validation_result.get("accepted", True):
            self.validation_stats["rejected"] += 1
            trigger = UpdateTrigger(
                type=TriggerType.VALIDATION_FEEDBACK,
                source="validation_rejection",
                confidence=0.9,
                details={
                    "validation_result": validation_result,
                    "reason": validation_result.get("reason", "Unknown reason")
                },
                timestamp=self._get_timestamp()
            )
            self.validation_stats["triggered_updates"] += 1
            return trigger
        
        # Check if validation was modified
        if validation_result.get("modified", False):
            self.validation_stats["modified"] += 1
            trigger = UpdateTrigger(
                type=TriggerType.VALIDATION_FEEDBACK,
                source="validation_modification",
                confidence=0.8,
                details={
                    "validation_result": validation_result,
                    "original_triple": validation_result.get("original_triple"),
                    "modified_triple": validation_result.get("triple")
                },
                timestamp=self._get_timestamp()
            )
            self.validation_stats["triggered_updates"] += 1
            return trigger
        
        # No trigger detected
        return None
    
    def _get_timestamp(self) -> str:
        """
        Get current timestamp in ISO format.
        
        Returns:
            Current timestamp string
        """
        return datetime.utcnow().isoformat()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about validation-based triggers.
        
        Returns:
            Trigger statistics
        """
        return self.validation_stats


class SchemaEvolutionTrigger:
    """
    Detects and processes schema evolution triggers.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 kb_llm = None):
        """
        Initialize the schema evolution trigger detector.
        
        Args:
            confidence_threshold: Confidence threshold for triggers
            kb_llm: Knowledge Builder LLM for schema suggestions
        """
        self.confidence_threshold = confidence_threshold
        self.kb_llm = kb_llm
        
        # Track schema evolution stats
        self.schema_stats = {
            "evolution_checks": 0,
            "detected_needs": 0,
            "triggered_updates": 0
        }
    
    def detect_schema_need(self, 
                          extractions: List[Dict[str, Any]], 
                          current_schema: Dict[str, Any]) -> Optional[UpdateTrigger]:
        """
        Detect if extractions indicate a need for schema evolution.
        
        Args:
            extractions: Latest knowledge extractions
            current_schema: Current schema definition
            
        Returns:
            Update trigger if need detected, None otherwise
        """
        self.schema_stats["evolution_checks"] += 1
        
        # Get schema constraints
        entity_types = set(current_schema.get("entity_types", []))
        relationship_types = set(current_schema.get("relationship_types", []))
        
        # Track potential schema violations
        unknown_entity_types = set()
        unknown_relationship_types = set()
        
        # Analyze extractions for schema violations
        for extraction in extractions:
            if "triples" in extraction:
                for triple in extraction["triples"]:
                    # Check entity types
                    if "subject_type" in triple and triple["subject_type"] not in entity_types:
                        unknown_entity_types.add(triple["subject_type"])
                    
                    if "object_type" in triple and triple["object_type"] not in entity_types and triple["predicate"].lower() not in ["type", "is_a"]:
                        unknown_entity_types.add(triple["object_type"])
                    
                    # Check relationship types
                    if triple["predicate"] not in relationship_types:
                        unknown_relationship_types.add(triple["predicate"])
        
        # If violations found, trigger schema update
        if unknown_entity_types or unknown_relationship_types:
            self.schema_stats["detected_needs"] += 1
            trigger = UpdateTrigger(
                type=TriggerType.SCHEMA_EVOLUTION,
                source="extraction_violations",
                confidence=0.8,
                details={
                    "unknown_entity_types": list(unknown_entity_types),
                    "unknown_relationship_types": list(unknown_relationship_types),
                    "current_schema": current_schema
                },
                timestamp=self._get_timestamp()
            )
            self.schema_stats["triggered_updates"] += 1
            return trigger
        
        # No need detected
        return None
    
    def suggest_schema_update(self, trigger: UpdateTrigger) -> Dict[str, Any]:
        """
        Suggest schema updates based on trigger information.
        
        Args:
            trigger: Schema evolution trigger
            
        Returns:
            Suggested schema updates
        """
        if not self.kb_llm:
            logger.error("KB LLM not available for schema suggestions")
            return {}
        
        # Get trigger details
        unknown_entity_types = trigger.details.get("unknown_entity_types", [])
        unknown_relationship_types = trigger.details.get("unknown_relationship_types", [])
        current_schema = trigger.details.get("current_schema", {})
        
        # Generate schema update prompt
        from reflexive_composition.llm2kg.prompt_templates import SchemaPrompts
        
        # Format the new information
        new_information = f"""
Unknown entity types detected: {', '.join(unknown_entity_types)}
Unknown relationship types detected: {', '.join(unknown_relationship_types)}

These types appear in recent knowledge extractions but are not part of the current schema.
"""
        
        # Generate schema evolution prompt
        prompt = SchemaPrompts.schema_evolution(current_schema, new_information)
        
        # Get schema update suggestions
        schema_suggestion = self.kb_llm.extract(prompt)
        
        # Process the result
        if isinstance(schema_suggestion, dict):
            return schema_suggestion
        else:
            # Handle error case
            logger.error(f"Failed to generate schema suggestion: {schema_suggestion}")
            return {
                "entity_types_to_add": unknown_entity_types,
                "relationship_types_to_add": unknown_relationship_types,
                "attributes_to_add": [],
                "reasoning": "Automatically suggested based on extraction violations"
            }
    
    def _get_timestamp(self) -> str:
        """
        Get current timestamp in ISO format.
        
        Returns:
            Current timestamp string
        """
        return datetime.utcnow().isoformat()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about schema evolution triggers.
        
        Returns:
            Trigger statistics
        """
        return self.schema_stats


class UpdateTriggerManager:
    """
    Manages all update triggers for knowledge graph maintenance.
    """
    
    def __init__(self, kb_llm = None):
        """
        Initialize the update trigger manager.
        
        Args:
            kb_llm: Knowledge Builder LLM for extraction and schema suggestions
        """
        self.kb_llm = kb_llm
        
        # Initialize trigger detectors
        self.query_trigger = QueryBasedTrigger(kb_llm=kb_llm)
        self.validation_trigger = ValidationFeedbackTrigger()
        self.schema_trigger = SchemaEvolutionTrigger(kb_llm=kb_llm)
        
        # Track all triggers
        self.triggers_history = []
    
    def detect_query_trigger(self, 
                            query: str, 
                            response: Dict[str, Any]) -> Optional[UpdateTrigger]:
        """
        Detect if a query response indicates a knowledge gap.
        
        Args:
            query: User query
            response: Generated response
            
        Returns:
            Update trigger if detected, None otherwise
        """
        trigger = self.query_trigger.detect_query_failure(query, response)
        
        if trigger:
            self.triggers_history.append(trigger)
            
        return trigger
    
    def detect_validation_trigger(self, 
                                validation_result: Dict[str, Any]) -> Optional[UpdateTrigger]:
        """
        Detect if validation feedback indicates a knowledge update need.
        
        Args:
            validation_result: Result from validator
            
        Returns:
            Update trigger if detected, None otherwise
        """
        trigger = self.validation_trigger.detect_validation_trigger(validation_result)
        
        if trigger:
            self.triggers_history.append(trigger)
            
        return trigger
    
    def detect_schema_need(self, 
                          extractions: List[Dict[str, Any]], 
                          current_schema: Dict[str, Any]) -> Optional[UpdateTrigger]:
        """
        Detect if extractions indicate a need for schema evolution.
        
        Args:
            extractions: Latest knowledge extractions
            current_schema: Current schema definition
            
        Returns:
            Update trigger if need detected, None otherwise
        """
        trigger = self.schema_trigger.detect_schema_need(extractions, current_schema)
        
        if trigger:
            self.triggers_history.append(trigger)
            
        return trigger
    
    def process_trigger(self, 
                       trigger: UpdateTrigger, 
                       knowledge_sources: List[str]) -> Dict[str, Any]:
        """
        Process an update trigger to extract new knowledge.
        
        Args:
            trigger: Update trigger
            knowledge_sources: List of source text to extract from
            
        Returns:
            Extracted knowledge
        """
        if trigger.type in [TriggerType.QUERY_FAILURE, TriggerType.CONTRADICTION]:
            return self.query_trigger.process_trigger(trigger, knowledge_sources)
        
        elif trigger.type == TriggerType.SCHEMA_EVOLUTION:
            return self.schema_trigger.suggest_schema_update(trigger)
        
        # Default empty response for unsupported trigger types
        return {"triples": []}
    
    def get_recent_triggers(self, 
                           limit: int = 10, 
                           trigger_type: Optional[TriggerType] = None) -> List[UpdateTrigger]:
        """
        Get recent update triggers.
        
        Args:
            limit: Maximum number of triggers to return
            trigger_type: Optional filter by trigger type
            
        Returns:
            List of recent triggers
        """
        if trigger_type:
            filtered_triggers = [t for t in self.triggers_history if t.type == trigger_type]
            return filtered_triggers[-limit:]
        else:
            return self.triggers_history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all update triggers.
        
        Returns:
            Trigger statistics
        """
        return {
            "query_triggers": self.query_trigger.get_stats(),
            "validation_triggers": self.validation_trigger.get_stats(),
            "schema_triggers": self.schema_trigger.get_stats(),
            "total_triggers": len(self.triggers_history)
        }
