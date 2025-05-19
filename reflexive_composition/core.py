# reflexive_composition/core.py
"""
Core components for the Reflexive Composition framework that enables 
bidirectional enhancement between Language Models and Knowledge Graphs.

The framework consists of three main components:
    1. LLM2KG: LLM-based knowledge extraction and graph construction
    2. HITL: Human-in-the-loop validation framework
    3. KG2LLM: Knowledge graph enhanced LLM inference

It also supports reflexive updaets, schema evolution, and
strategic validation as described in the Reflexive Composition methodology
"""
import logging
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from reflexive_composition.utils.llm_utils import extract_text

logger = logging.getLogger(__name__)

class ReflexiveComposition:
    """
    Main framework class that orchestrates the reflexive interaction between
    LLMs, knowledge graphs, and human validators.
    """
    
    def __init__(self, 
                 kb_llm_config: Dict[str, Any],
                 target_llm_config: Dict[str, Any],
                 kg_config: Dict[str, Any],
                 hitl_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Reflexive Composition framework.
        
        Args:
            kb_llm_config: Configuration for the Knowledge Builder LLM
            target_llm_config: Configuration for the Target LLM
            kg_config: Configuration for the Knowledge Graph
            hitl_config: Configuration for the Human-in-the-loop component
        """
        self.kb_llm = None  # Will be initialized with LLM for extraction
        self.target_llm = None  # Will be initialized with LLM for generation
        self.kg = None  # Will hold the knowledge graph
        self.validator = None  # Will hold the HITL validation component
        self.update_trigger_manager = None  # Will manage update triggers

        self.kg_config = kg_config
        self.hitl_config = hitl_config
        
        # Initialize components
        self._init_llm2kg(kb_llm_config)
        self._init_kg2llm(target_llm_config)
        self._init_knowledge_graph(kg_config)
        
        from reflexive_composition.kg2llm.prompt_builder import PromptBuilder
        self.prompt_builder = PromptBuilder()

        if hitl_config:
            self._init_hitl(hitl_config)
        else:
            logger.warning("No HITL configuration provided. Running in automated mode.")

        # Initialize update trigger manager
        self._init_update_triggers()
    
    def _init_llm2kg(self, config: Dict[str, Any]) -> None:
        """Initialize the LLM2KG component."""
        from reflexive_composition.llm2kg import KnowledgeBuilderLLM
        self.kb_llm = KnowledgeBuilderLLM(**config)

        # Initialize schema manager for evolution
        from reflexive_composition.llm2kg.schema_evolution import SchemaManager
        self.schema_manager = SchemaManager(kb_llm=self.kb_llm)
    
    def _init_kg2llm(self, config: Dict[str, Any]) -> None:
        """Initialize the KG2LLM component."""
        from reflexive_composition.kg2llm import TargetLLM
        self.target_llm = TargetLLM(**config)
    
    def _init_knowledge_graph(self, config: Dict[str, Any]) -> None:
        """Initialize the Knowledge Graph component."""
        from reflexive_composition.knowledge_graph import KnowledgeGraph
        self.kg = KnowledgeGraph(**config)
    
    def _init_hitl(self, config: Dict[str, Any]) -> None:
        """Initialize the HITL component."""
        from reflexive_composition.hitl import Validator
        self.validator = Validator(config=config)
    
    def _init_update_triggers(self) -> None:
        """Initialize the update trigger manager."""
        from reflexive_composition.knowledge_graph.update_triggers import UpdateTriggerManager
        self.update_trigger_manager = UpdateTriggerManager(kb_llm=self.kb_llm)
    
    def extract_knowledge(self, 
                        source_text: str, 
                        schema: Optional[Dict[str, Any]] = None,
                        confidence_threshold: float = 0.7,
                        extraction_type: Optional[str] = None,
                        domain: Optional[str] = None,
                        auto_detect: bool = True,
                        interactive: bool = True,
                        debug: bool = False) -> Dict[str, Any]:
        """
        Extract knowledge from source text using the KB-LLM.
        
        Args:
            source_text: The text to extract knowledge from
            schema: Optional schema to guide extraction
            confidence_threshold: Threshold for automatic acceptance
            extraction_type: Type of extraction ("general", "temporal", "domain")
            domain: Domain for domain-specific extraction
            auto_detect: Whether to auto-detect extraction type and domain
            debug: Enable debug output
            
        Returns:
            Extracted knowledge in structured format
        """
        # Auto-detect extraction type and domain if requested
        if auto_detect and extraction_type is None:
            from reflexive_composition.utils.text_utils import suggest_extraction_type
            suggestion = suggest_extraction_type(source_text)
            extraction_type = suggestion['extraction_type']
            
            # Only override domain if not explicitly provided
            if domain is None:
                domain = suggestion['domain']
                
            if debug:
                print(f"DEBUG: Auto-detected extraction type: {extraction_type}")
                if domain:
                    print(f"DEBUG: Auto-detected domain: {domain}")
        
        # Default to general extraction if not specified
        extraction_type = extraction_type or "general"
                
        # Extract candidate triples
        if debug:
            print(f"DEBUG: Extracting knowledge from text of length {len(source_text)} characters")
            print(f"DEBUG: Using KB-LLM: {getattr(self.kb_llm, 'model_name', 'Unknown')} ({getattr(self.kb_llm, 'model_provider', 'Unknown')})")
            print(f"DEBUG: Extraction type: {extraction_type}")
        
        # Generate extraction prompt
        from reflexive_composition.llm2kg.prompt_templates import (
            get_extraction_prompt, 
            get_temporal_extraction_prompt,
            get_domain_specific_extraction_prompt
        )
        
        if extraction_type == "temporal":
            prompt = get_temporal_extraction_prompt(source_text, schema)
        elif extraction_type == "domain" and domain:
            prompt = get_domain_specific_extraction_prompt(source_text, domain, schema)
        else:
            # Default to general extraction
            prompt = get_extraction_prompt(source_text, schema)
            
        if debug:
            print(f"\n=== Prompt sent to KB LLM ===\n{prompt}\n")
        
        candidate_triples = self.kb_llm.extract(prompt, schema)
        
        if debug:
            print(f"DEBUG: Extraction result type: {type(candidate_triples)}")
            import json
            triple_dump = json.dumps(candidate_triples, indent=2, default=str)
            print(f"DEBUG: Extraction result: {triple_dump}")
        
        # Handle case where extraction result is not a dictionary
        if not isinstance(candidate_triples, dict):
            if debug:
                print(f"DEBUG: Invalid extraction result format. Expected dict, got {type(candidate_triples)}")
            
            # If it's a string, try to parse it as JSON
            if isinstance(candidate_triples, str):
                try:
                    import json
                    parsed_result = json.loads(candidate_triples)
                    if debug:
                        print(f"DEBUG: Parsed string as JSON: {json.dumps(parsed_result, indent=2)}")
                    candidate_triples = parsed_result
                except json.JSONDecodeError:
                    if debug:
                        print(f"DEBUG: Failed to parse string as JSON")
                    return {"triples": []}
            else:
                return {"triples": []}
        
        # Ensure 'triples' key exists in the dictionary
        if 'triples' not in candidate_triples:
            if debug:
                print(f"DEBUG: 'triples' key not found in extraction result")
            
            # If the entire dict is structured like a single triple, wrap it
            if all(k in candidate_triples for k in ['subject', 'predicate', 'object']):
                if debug:
                    print(f"DEBUG: Converting single triple structure to list")
                candidate_triples = {'triples': [candidate_triples]}
            else:
                if debug:
                    print(f"DEBUG: Returning empty triples list")
                return {'triples': []}
        
        if debug:
            print(f"DEBUG: Processing {len(candidate_triples.get('triples', []))} candidate triples")
        
        # Process extracted triples
        if self.validator:
            validated_triples = []
            for i, triple in enumerate(candidate_triples.get('triples', [])):
                if debug:
                    print(f"DEBUG: Processing triple {i+1}: {triple}")
                
                # Ensure triple is a dictionary
                if not isinstance(triple, dict):
                    if debug:
                        print(f"DEBUG: Invalid triple format. Expected dict, got {type(triple)}")
                    continue
                
                confidence = triple.get('confidence', 0)
                if debug:
                    print(f"DEBUG: Triple confidence: {confidence}, threshold: {confidence_threshold}")
                
                if confidence >= confidence_threshold:
                    if debug:
                        print(f"DEBUG: Auto-accepting triple (confidence above threshold)")
                    validated_triples.append(triple)
                else:
                    if debug:
                        print(f"DEBUG: Routing triple to validator")
                    # Route to validator
                    validation_result = self.validator.validate_triple(
                        triple, source_text, self.knowledge_graph, interactive=interactive
                    )
                    if validation_result.get('accepted', False):
                        if debug:
                            print(f"DEBUG: Validator accepted triple")
                        validated_triples.append(validation_result.get('triple', triple))
                    elif debug:
                        print(f"DEBUG: Validator rejected triple")
            
            if debug:
                print(f"DEBUG: Returning {len(validated_triples)} validated triples")
            
            return {'triples': validated_triples}
        
        # Without validator, return all triples above threshold
        filtered_triples = []
        for i, triple in enumerate(candidate_triples.get('triples', [])):
            if debug:
                print(f"DEBUG: Processing triple {i+1} without validator")
            
            # Ensure triple is a dictionary
            if not isinstance(triple, dict):
                if debug:
                    print(f"DEBUG: Invalid triple format. Expected dict, got {type(triple)}")
                continue
            
            confidence = triple.get('confidence', 0)
            if confidence >= confidence_threshold:
                if debug:
                    print(f"DEBUG: Accepting triple (confidence {confidence} above threshold {confidence_threshold})")
                filtered_triples.append(triple)
            elif debug:
                print(f"DEBUG: Rejecting triple (confidence {confidence} below threshold {confidence_threshold})")
        
        if debug:
            print(f"DEBUG: Returning {len(filtered_triples)} filtered triples")
        
        return {'triples': filtered_triples}
    
    def update_knowledge_graph(self, triples: List[Dict[str, Any]]) -> bool:
        """
        Update the knowledge graph with validated triples.
        
        Args:
            triples: List of validated triples to add to the KG
            
        Returns:
            Success status
        """
        return self.kg.add_triples(triples)
    
    def update_schema(self,
                      schema_update: Dict[str, Any],
                      validate: bool = True) -> Dict[str, Any]:
        """
        Update the knowledge graph schema.
        
        Args:
            schema_update: Proposed schema update
            validate: Whether to validate the update with HITL
            
        Returns:
            Updated schema
        """
        current_schema = self.knowledge_graph.schema
        
        if validate and self.validator:
            # Route schema update to validator
            validation_result = self.validator.validate_schema_update(
                schema_update, current_schema
            )
            
            if not validation_result.get('accepted', False):
                logger.warning(f"Schema update rejected by validator: {validation_result.get('reason', 'Unknown reason')}")
                return current_schema
            
            # Use validated schema update
            if validation_result.get('modified', False):
                schema_update = validation_result.get('schema_update', schema_update)
        
        # Process entity types to add
        entity_types_to_add = schema_update.get('entity_types_to_add', [])
        relationship_types_to_add = schema_update.get('relationship_types_to_add', [])
        
        # Create updated schema
        updated_schema = current_schema.copy()
        
        # Add new entity types
        if 'entity_types' not in updated_schema:
            updated_schema['entity_types'] = []
        
        for entity_type in entity_types_to_add:
            if entity_type not in updated_schema['entity_types']:
                updated_schema['entity_types'].append(entity_type)
        
        # Add new relationship types
        if 'relationship_types' not in updated_schema:
            updated_schema['relationship_types'] = []
        
        for rel_type in relationship_types_to_add:
            if rel_type not in updated_schema['relationship_types']:
                updated_schema['relationship_types'].append(rel_type)
        
        # Update version
        updated_schema['version'] = current_schema.get('version', 0) + 1
        
        # Update the knowledge graph schema
        success = self.kg.update_schema(updated_schema)
        
        if success:
            logger.info(f"Schema updated to version {updated_schema.get('version')}")
            return updated_schema
        else:
            logger.error("Failed to update schema")
            return current_schema
    
    def suggest_schema(self, domain_description: str) -> Dict[str, Any]:
        """
        Suggest a knowledge graph schema for a domain.
        
        Args:
            domain_description: Description of the domain
            
        Returns:
            Suggested schema
        """
        if hasattr(self, 'schema_manager'):
            return self.schema_manager.suggest_schema_from_description(
                domain_description, self.kb_llm
            )
        else:
            # Fallback if schema manager not available
            from reflexive_composition.llm2kg.prompt_templates import SchemaPrompts
            
            # Generate schema prompt
            prompt = SchemaPrompts.schema_generation(domain_description)
            
            # Get schema suggestion from KB-LLM
            schema_suggestion = self.kb_llm.extract(prompt)
            
            if isinstance(schema_suggestion, dict):
                # Add version number
                schema_suggestion["version"] = 1
                return schema_suggestion
            else:
                logger.error(f"Failed to generate schema suggestion")
                return {
                    "entity_types": [],
                    "relationship_types": [],
                    "version": 1
                }
    
    def generate_response(self,
                          query: str,
                          grounded: bool = True,
                          context_label: str = "Facts",
                          template: Optional[str] = None,
                          max_context_items: int = 10) -> Dict[str, Any]:
        """
        Generate a response using the target LLM, optionally grounded in the knowledge graph.
        
        Args:
            query: User query
            grounded: Whether to ground the response in the knowledge graph
            context_label: Label for the context block
            template: Optional prompt template
            max_context_items: Maximum number of context items to include
            
        Returns:
            Generated response
        """
        if grounded:
            # Get relevant context from knowledge graph
            context = self.kg.retrieve_context(query, max_context_items)
            triples = context.get("triples", [])
        else:
            triples = []
        
        # Build prompt with context
        prompt, meta = self.prompt_builder.build_prompt(
            query=query,
            context=triples,
            context_label=context_label,
            template=template
        )
        
        # Generate response with context
        response = self.target_llm.generate(prompt)
        
        # Check for query-based update triggers
        if self.update_trigger_manager:
            trigger = self.update_trigger_manager.detect_query_trigger(query, response)
            if trigger:
                logger.info(f"Query update trigger detected: {trigger.source}")
        
        # Check for contradictions
        from reflexive_composition.kg2llm.contradiction_detector import ContradictionDetector
        contradictions = ContradictionDetector().detect_contradictions(response.get("text", ""), triples)
        
        if contradictions and self.validator:
            # Route contradictions to validator
            validated_response = self.validator.validate_response(
                response, contradictions, {"triples": triples}
            )
            
            # If modified by validator, use the validated response
            if validated_response.get("modified", False):
                response["text"] = validated_response.get("corrected_text", response["text"])
                response["meta"]["validator_notes"] = validated_response.get("notes", "")
                response["meta"]["was_validated"] = True
        
        return response
    
    def process_reflexive_update(self, 
                                query: str, 
                                response: Dict[str, Any],
                                knowledge_sources: List[str]) -> Dict[str, Any]:
        """
        Process reflexive updates based on query and response.
        
        Args:
            query: User query
            response: Generated response
            knowledge_sources: List of source text for extraction
            
        Returns:
            Update results
        """
        if not self.update_trigger_manager:
            logger.error("Update trigger manager not available for reflexive updates")
            return {"status": "error", "reason": "Update trigger manager not available"}
        
        # Detect query-based triggers
        trigger = self.update_trigger_manager.detect_query_trigger(query, response)
        
        if not trigger:
            logger.info("No update triggers detected")
            return {"status": "no_update_needed"}
        
        # Process trigger to extract new knowledge
        extracted = self.update_trigger_manager.process_trigger(trigger, knowledge_sources)
        
        if not extracted or not extracted.get("triples"):
            logger.info("No new knowledge extracted from trigger")
            return {"status": "no_knowledge_extracted"}
        
        # Validate extracted knowledge
        if self.validator:
            validated_triples = []
            
            for triple in extracted.get("triples", []):
                # Get most relevant source text
                source_text = knowledge_sources[0] if knowledge_sources else ""
                
                # Validate triple
                validation_result = self.validator.validate_triple(
                    triple, source_text, self.kg
                )
                
                if validation_result.get("accepted", False):
                    validated_triples.append(validation_result.get("triple", triple))
            
            # Update knowledge graph with validated triples
            if validated_triples:
                success = self.kg.add_triples(validated_triples)
                
                if success:
                    logger.info(f"Updated knowledge graph with {len(validated_triples)} triples from reflexive update")
                    return {
                        "status": "success",
                        "triples_added": len(validated_triples),
                        "trigger_type": str(trigger.type)
                    }
                else:
                    logger.error("Failed to update knowledge graph")
                    return {"status": "error", "reason": "Failed to update knowledge graph"}
            else:
                logger.info("No triples validated for reflexive update")
                return {"status": "no_triples_validated"}
        else:
            # Without validator, update directly with extracted triples
            success = self.kg.add_triples(extracted.get("triples", []))
            
            if success:
                logger.info(f"Updated knowledge graph with {len(extracted.get('triples', []))} triples from reflexive update")
                return {
                    "status": "success",
                    "triples_added": len(extracted.get("triples", [])),
                    "trigger_type": str(trigger.type)
                }
            else:
                logger.error("Failed to update knowledge graph")
                return {"status": "error", "reason": "Failed to update knowledge graph"}
    
    def _is_temporal_text(self, text: str) -> bool:
        """
        Determine if text contains significant temporal information.
        
        Args:
            text: Source text
            
        Returns:
            True if temporal text, False otherwise
        """
        import re
        
        # Check for date patterns
        date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY or DD/MM/YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'  # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check for temporal keywords
        temporal_keywords = [
            'yesterday', 'today', 'tomorrow', 'last week', 'next week',
            'last month', 'next month', 'last year', 'next year',
            'recent', 'recently', 'latest', 'current', 'upcoming',
            'schedule', 'timeline', 'history'
        ]
        
        for keyword in temporal_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                return True
        
        return False
