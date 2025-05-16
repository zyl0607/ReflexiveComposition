# reflexive_composition/core.py
"""
Core components for the Reflexive Composition framework that enables 
bidirectional enhancement between Language Models and Knowledge Graphs.

The framework consists of three main components:
1. LLM2KG: LLM-based knowledge extraction and graph construction
2. HITL: Human-in-the-loop validation framework
3. KG2LLM: Knowledge graph enhanced LLM inference
"""
import logging
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
        self.knowledge_graph = None  # Will hold the knowledge graph
        self.validator = None  # Will hold the HITL validation component
        
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
    
    def _init_llm2kg(self, config: Dict[str, Any]) -> None:
        """Initialize the LLM2KG component."""
        from reflexive_composition.llm2kg import KnowledgeBuilderLLM
        self.kb_llm = KnowledgeBuilderLLM(**config)
    
    def _init_kg2llm(self, config: Dict[str, Any]) -> None:
        """Initialize the KG2LLM component."""
        from reflexive_composition.kg2llm import TargetLLM
        self.target_llm = TargetLLM(**config)
    
    def _init_knowledge_graph(self, config: Dict[str, Any]) -> None:
        """Initialize the Knowledge Graph component."""
        from reflexive_composition.knowledge_graph import KnowledgeGraph
        self.knowledge_graph = KnowledgeGraph(**config)
    
    def _init_hitl(self, config: Dict[str, Any]) -> None:
        """Initialize the HITL component."""
        from reflexive_composition.hitl import Validator
        self.validator = Validator(config=config)
    
    def extract_knowledge(self, 
                        source_text: str, 
                        schema: Optional[Dict[str, Any]] = None,
                        confidence_threshold: float = 0.7,
                        debug: bool = False) -> Dict[str, Any]:
        """
        Extract knowledge from source text using the KB-LLM.
        
        Args:
            source_text: The text to extract knowledge from
            schema: Optional schema to guide extraction
            confidence_threshold: Threshold for automatic acceptance
            debug: Enable debug output
            
        Returns:
            Extracted knowledge in structured format
        """
        # Extract candidate triples
        if debug:
            print(f"DEBUG: Extracting knowledge from text of length {len(source_text)} characters")
            print(f"DEBUG: Using KB-LLM: {getattr(self.kb_llm, 'model_name', 'Unknown')} ({getattr(self.kb_llm, 'model_provider', 'Unknown')})")
        
        candidate_triples = self.kb_llm.extract(source_text, schema)
        
        if debug:
            print(f"DEBUG: Extraction result type: {type(candidate_triples)}")
            print(f"DEBUG: Extraction result: {json.dumps(candidate_triples, indent=2, default=str)}")
        
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
        
        # Rest of the method remains the same, but with added debug prints
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
                        triple, source_text, self.knowledge_graph
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
        return self.knowledge_graph.add_triples(triples)
    
    def generate_response(self,
                          query: str,
                          grounded: bool = True,
                          context_label: str = "Facts",
                          template: Optional[str] = None,
                          max_context_items: int = 10) -> Dict[str, Any]:
        """
        Generate a response using the target LLM, optionally grounded in the knowledge graph.
        Returns:
            LLM output string or dictionary
        """
        triples = self.knowledge_graph.get_triples() if grounded else []
        print(">>> DEBUG: triples being passed to prompt builder:", triples)
        if max_context_items is not None:
            triples = triples[:max_context_items]

        prompt, meta = self.prompt_builder.build_prompt(query=query,
                                                        context=triples,
                                                        context_label=context_label,
                                                        template=template)
        print(">>> DEBUG: prompt:", prompt)
        # Generate response with context
        response = self.target_llm.generate(prompt)
        
        # Check for contradictions
        contradictions = self._detect_contradictions(response, triples)
        
        if contradictions and self.validator:
            # Route contradictions to validator
            validated_response = self.validator.validate_response(
                response, contradictions, triples
            )
            return validated_response
        
        return response
    
    def _detect_contradictions(self, 
                              response: Dict[str, Any], 
                              context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect contradictions between response and KG context.
        
        Args:
            response: The generated response
            context: The KG context used for generation
            
        Returns:
            List of detected contradictions
        """
        # This would implement contradiction detection logic
        # For now, returning empty list as placeholder
        return []
    
    def reflexive_update(self, 
                         query: str, 
                         response: Dict[str, Any],
                         feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform a reflexive update based on feedback.
        
        This closes the loop by extracting knowledge from responses and feedback,
        validating it, and updating the knowledge graph.
        
        Args:
            query: The original query
            response: The generated response
            feedback: Optional feedback (e.g., corrections, ratings)
            
        Returns:
            Update statistics
        """
        update_stats = {
            'extracted': 0,
            'validated': 0,
            'added_to_kg': 0,
        }
        
        # Extract knowledge from response
        extracted = self.kb_llm.extract(response.get('text', ''))
        update_stats['extracted'] = len(extracted.get('triples', []))
        
        # If feedback is provided, use it for validation
        if feedback and self.validator:
            validated = self.validator.validate_with_feedback(
                extracted.get('triples', []), feedback
            )
            update_stats['validated'] = len(validated.get('triples', []))
            
            # Update KG with validated triples
            if validated.get('triples'):
                success = self.knowledge_graph.add_triples(validated['triples'])
                if success:
                    update_stats['added_to_kg'] = len(validated['triples'])
        
        return update_stats