# reflexive_composition/hitl/interface.py
"""
Validation interfaces for human-in-the-loop interactions.

These interfaces handle the presentation of validation tasks to human
validators and collect their decisions.
"""

import logging
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class ValidationInterface:
    """
    Base class for validation interfaces.
    
    This abstract class defines the interface for presenting validation
    tasks to human validators and collecting their decisions.
    """
    
    def present_triple_validation(self, 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Present a triple validation task to a human validator.
        
        Args:
            context: Validation context
            
        Returns:
            Validation decision
        """
        raise NotImplementedError("Subclasses must implement present_triple_validation")
    
    def present_schema_validation(self, 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Present a schema validation task to a human validator.
        
        Args:
            context: Validation context
            
        Returns:
            Validation decision
        """
        raise NotImplementedError("Subclasses must implement present_schema_validation")
    
    def present_response_validation(self, 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Present a response validation task to a human validator.
        
        Args:
            context: Validation context
            
        Returns:
            Validation decision
        """
        raise NotImplementedError("Subclasses must implement present_response_validation")


class ConsoleValidationInterface(ValidationInterface):
    """
    Console-based validation interface.
    
    This interface presents validation tasks via the console and
    collects decisions via command line input.
    """
    
    def present_triple_validation(self, 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Present a triple validation task via the console.
        
        Args:
            context: Validation context
            
        Returns:
            Validation decision
        """
        triple = context.get("triple", {})
        source_text = context.get("source_text", "")
        contradiction = context.get("contradiction")
        
        print("\n=== TRIPLE VALIDATION ===")
        print(f"Subject: {triple.get('subject', '')}")
        print(f"Predicate: {triple.get('predicate', '')}")
        print(f"Object: {triple.get('object', '')}")
        print(f"Confidence: {triple.get('confidence', 0.0)}")
        
        print("\nSource Text:")
        print(source_text)
        
        if contradiction:
            print("\nContradiction Detected:")
            existing = contradiction.get("existing_triple", {})
            print(f"Existing: {existing.get('subject', '')} - {existing.get('predicate', '')} - {existing.get('object', '')}")
            print(f"Reason: {contradiction.get('reason', '')}")
        
        print("\nOptions:")
        print("1. Accept as is")
        print("2. Modify and accept")
        print("3. Reject")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            return {
                "accepted": True,
                "modified": False,
                "triple": triple,
                "validation_type": "human_accept",
                "notes": input("Optional notes: ")
            }
        elif choice == "2":
            # Collect modifications
            print("\nEnter modifications:")
            subject = input(f"Subject [{triple.get('subject', '')}]: ") or triple.get('subject', '')
            predicate = input(f"Predicate [{triple.get('predicate', '')}]: ") or triple.get('predicate', '')
            obj = input(f"Object [{triple.get('object', '')}]: ") or triple.get('object', '')
            
            modified_triple = {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "confidence": triple.get("confidence", 0.0)
            }
            
            return {
                "accepted": True,
                "modified": True,
                "triple": modified_triple,
                "original_triple": triple,
                "validation_type": "human_modify",
                "notes": input("Optional notes: ")
            }
        else:
            return {
                "accepted": False,
                "modified": False,
                "triple": triple,
                "validation_type": "human_reject",
                "reason": input("Reason for rejection: ")
            }
    
    def present_schema_validation(self, 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Present a schema validation task via the console.
        
        Args:
            context: Validation context
            
        Returns:
            Validation decision
        """
        schema_update = context.get("schema_update", {})
        current_schema = context.get("current_schema", {})
        
        print("\n=== SCHEMA VALIDATION ===")
        
        # Display entity type updates
        entity_types = schema_update.get("entity_types", [])
        if entity_types:
            print("\nProposed Entity Types:")
            for entity_type in entity_types:
                print(f"- {entity_type.get('type')}")
                print(f"  Confidence: {entity_type.get('confidence', 0.0)}")
                print(f"  Sources: {len(entity_type.get('source_extractions', []))}")
        
        # Display relationship type updates
        relationship_types = schema_update.get("relationship_types", [])
        if relationship_types:
            print("\nProposed Relationship Types:")
            for rel_type in relationship_types:
                print(f"- {rel_type.get('type')}")
                print(f"  Confidence: {rel_type.get('confidence', 0.0)}")
                print(f"  Sources: {len(rel_type.get('source_extractions', []))}")
        
        print("\nCurrent Schema:")
        print(f"- Entity Types: {', '.join(current_schema.get('entity_types', []))}")
        print(f"- Relationship Types: {', '.join(current_schema.get('relationship_types', []))}")
        
        print("\nOptions:")
        print("1. Accept all updates")
        print("2. Selectively accept updates")
        print("3. Reject all updates")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            return {
                "accepted": True,
                "modified": False,
                "schema_update": schema_update,
                "validation_type": "human_accept_all",
                "notes": input("Optional notes: ")
            }
        elif choice == "2":
            # Selective acceptance
            accepted_entity_types = []
            accepted_relationship_types = []
            
            # Process entity types
            for entity_type in entity_types:
                type_name = entity_type.get('type')
                accept = input(f"Accept entity type '{type_name}'? (y/n): ").lower() == 'y'
                if accept:
                    accepted_entity_types.append(entity_type)
            
            # Process relationship types
            for rel_type in relationship_types:
                type_name = rel_type.get('type')
                accept = input(f"Accept relationship type '{type_name}'? (y/n): ").lower() == 'y'
                if accept:
                    accepted_relationship_types.append(rel_type)
            
            modified_update = {
                "entity_types": accepted_entity_types,
                "relationship_types": accepted_relationship_types
            }
            
            return {
                "accepted": len(accepted_entity_types) > 0 or len(accepted_relationship_types) > 0,
                "modified": True,
                "schema_update": modified_update,
                "original_update": schema_update,
                "validation_type": "human_selective_accept",
                "notes": input("Optional notes: ")
            }
        else:
            return {
                "accepted": False,
                "modified": False,
                "schema_update": schema_update,
                "validation_type": "human_reject",
                "reason": input("Reason for rejection: ")
            }
    
    def present_response_validation(self, 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Present a response validation task via the console.
        
        Args:
            context: Validation context
            
        Returns:
            Validation decision
        """
        response = context.get("response", {})
        contradictions = context.get("contradictions", [])
        kg_context = context.get("kg_context", {})
        
        print("\n=== RESPONSE VALIDATION ===")
        
        print("\nGenerated Response:")
        print(response.get("text", ""))
        
        print("\nDetected Contradictions:")
        for i, contradiction in enumerate(contradictions, 1):
            print(f"\nContradiction {i}:")
            print(f"- Statement: {contradiction.get('statement', '')}")
            print(f"- Conflicting Fact: {contradiction.get('conflicting_fact', '')}")
        
        print("\nKG Context Used:")
        for i, triple in enumerate(kg_context.get("triples", []), 1):
            print(f"{i}. {triple.get('subject', '')} - {triple.get('predicate', '')} - {triple.get('object', '')}")
        
        print("\nOptions:")
        print("1. Accept response as is")
        print("2. Correct the response")
        print("3. Reject the response")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            return {
                "accepted": True,
                "modified": False,
                "response": response,
                "validation_type": "human_accept",
                "notes": input("Optional notes: ")
            }
        elif choice == "2":
            print("\nCurrent response:")
            print(response.get("text", ""))
            
            corrected_text = input("\nEnter corrected response (or press enter to use multiline mode):\n")
            
            # Support multiline input if needed
            if not corrected_text:
                print("Enter corrected response (type 'END' on a new line when finished):")
                lines = []
                while True:
                    line = input()
                    if line == "END":
                        break
                    lines.append(line)
                corrected_text = "\n".join(lines)
            
            return {
                "accepted": True,
                "modified": True,
                "response": response,
                "corrected_text": corrected_text,
                "validation_type": "human_correct",
                "notes": input("Optional notes: ")
            }
        else:
            return {
                "accepted": False,
                "modified": False,
                "response": response,
                "validation_type": "human_reject",
                "reason": input("Reason for rejection: ")
            }


class WebValidationInterface(ValidationInterface):
    """
    Web-based validation interface.
    
    This interface presents validation tasks via a web interface and
    collects decisions via form submissions.
    
    Note: This is a placeholder implementation. A real implementation
    would integrate with a web application framework.
    """
    
    def __init__(self, api_url: Optional[str] = None):
        """
        Initialize the web validation interface.
        
        Args:
            api_url: URL of the validation API
        """
        self.api_url = api_url or "http://localhost:8000/api/validate"
        logger.info(f"Initialized web validation interface with API URL: {self.api_url}")
    
    def present_triple_validation(self, 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Present a triple validation task via the web interface.
        
        Args:
            context: Validation context
            
        Returns:
            Validation decision
        """
        # In a real implementation, this would send the context to the web API
        # and wait for a response. For now, we'll simulate a response.
        logger.info(f"Web validation interface would present triple validation: {context.get('triple')}")
        
        # Simulate a validation decision
        return {
            "accepted": True,
            "modified": False,
            "triple": context.get("triple"),
            "validation_type": "web_accept",
            "notes": "Validated via web interface (simulated)"
        }
    
    def present_schema_validation(self, 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Present a schema validation task via the web interface.
        
        Args:
            context: Validation context
            
        Returns:
            Validation decision
        """
        # In a real implementation, this would send the context to the web API
        # and wait for a response. For now, we'll simulate a response.
        logger.info(f"Web validation interface would present schema validation with {len(context.get('schema_update', {}).get('entity_types', []))} entity types")
        
        # Simulate a validation decision
        return {
            "accepted": True,
            "modified": False,
            "schema_update": context.get("schema_update"),
            "validation_type": "web_accept",
            "notes": "Validated via web interface (simulated)"
        }
    
    def present_response_validation(self, 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Present a response validation task via the web interface.
        
        Args:
            context: Validation context
            
        Returns:
            Validation decision
        """
        # In a real implementation, this would send the context to the web API
        # and wait for a response. For now, we'll simulate a response.
        logger.info(f"Web validation interface would present response validation with {len(context.get('contradictions', []))} contradictions")
        
        # Simulate a validation decision
        return {
            "accepted": True,
            "modified": False,
            "response": context.get("response"),
            "validation_type": "web_accept",
            "notes": "Validated via web interface (simulated)"
        }