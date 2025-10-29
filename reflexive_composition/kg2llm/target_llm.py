# reflexive_composition/kg2llm/target_llm.py
"""
Target LLM component for knowledge graph enhanced inference.

This module handles the interaction with LLMs for generating responses
using knowledge graph context.
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from reflexive_composition.utils.llm_utils import extract_text, attach_prompt_stats

logger = logging.getLogger(__name__)

class TargetLLM:
    """
    Target LLM for knowledge graph enhanced inference.
    
    This class handles the generation of responses using LLMs with
    knowledge graph context integration.
    """
    
    def __init__(self, 
                 model_name: str,
                 api_key: Optional[str] = None,
                 model_provider: str = "openai",
                 max_tokens: int = 1000,
                 temperature: float = 0.7,
                 custom_llm_client: Optional[Any] = None):
        """
        Initialize the Target LLM.
        
        Args:
            model_name: Name/identifier of the LLM
            api_key: API key for accessing the LLM service
            model_provider: Provider of the LLM (openai, anthropic, etc.)
            max_tokens: Maximum tokens for LLM generation
            temperature: LLM temperature parameter
            custom_llm_client: Optional custom LLM client implementation
        """
        self.model_name = model_name
        self.api_key = api_key
        self.model_provider = model_provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize the LLM client
        self.llm_client = custom_llm_client or self._init_llm_client()
        
        # Initialize prompt builder
        from .prompt_builder import PromptBuilder
        self.prompt_builder = PromptBuilder()
        
        # Initialize contradiction detector
        from .contradiction_detector import ContradictionDetector
        self.contradiction_detector = ContradictionDetector()
    
    def _init_llm_client(self) -> Any:
        """
        Initialize the appropriate LLM client based on the provider.
        
        Returns:
            Initialized LLM client
        """
        if self.model_provider == "openai":
            try:
                import openai
                # Support new OpenAI API structure
                if hasattr(openai, 'OpenAI'):
                    client = openai.OpenAI(
                        api_key=self.api_key,
                        base_url=os.environ.get("OPENAI_BASE_URL", None)
                    )
                    return client
                else:
                    # Old API structure
                    openai.api_key = self.api_key
                    return openai
            except ImportError:
                logger.error("OpenAI package not found. Please install with 'pip install openai'")
                raise
        
        elif self.model_provider == "anthropic":
            try:
                import anthropic
                return anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.error("Anthropic package not found. Please install with 'pip install anthropic'")
                raise
        
        elif self.model_provider == "google":  # Add Google Gemini support
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                return genai
            except ImportError:
                logger.error("Google GenerativeAI package not found. Please install with 'pip install google-generativeai'")
                raise

        elif self.model_provider == "huggingface":
            try:
                from transformers import pipeline
                return pipeline("text-generation", model=self.model_name)
            except ImportError:
                logger.error("Transformers package not found. Please install with 'pip install transformers'")
                raise
        
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")
    
    def generate(self, 
                query: str, 
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a response using the LLM with knowledge graph context.
        
        Args:
            query: User query
            context: Knowledge graph context
            
        Returns:
            Generated response
        """
        # Prepare response metadata
        response_meta = {
            "model": self.model_name,
            "provider": self.model_provider,
            "query": query
        }
        
        # The prompt is assumed to have already been constructed
        prompt = query  # Assume the prompt has already been constructed

        # Generate using the LLM
        try:
            if self.model_provider == "openai":
                llm_response = self._generate_with_openai(prompt)
            elif self.model_provider == "anthropic":
                llm_response = self._generate_with_anthropic(prompt)
            elif self.model_provider == "google":  # Add Google provider case
                llm_response = self._generate_with_google(prompt)
            elif self.model_provider == "huggingface":
                llm_response = self._generate_with_huggingface(prompt)
            else:
                raise ValueError(f"Unsupported model provider for generation: {self.model_provider}")
            
            # Extract the generated text
            if isinstance(llm_response, str):
                generated_text = llm_response
            else:
                # Handle different response formats
                generated_text = self._extract_response_text(llm_response)
            
            # Check for contradictions
            if context:
                contradictions = self.contradiction_detector.detect_contradictions(
                    generated_text, context
                )
                response_meta["contradictions"] = contradictions
                response_meta["has_contradictions"] = len(contradictions) > 0
            else:
                response_meta["has_contradictions"] = False
            
            attach_prompt_stats(prompt, response_meta)
            
            # Prepare the final response
            response = {
                "text": generated_text,
                "meta": response_meta
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            
            # Return error response
            return {
                "text": f"Error generating response: {str(e)}",
                "meta": {
                    **response_meta,
                    "error": str(e),
                    "has_contradictions": False
                }
            }
    
    def _generate_with_openai(self, prompt: str) -> Any:
        """
        Generate text using OpenAI's API (compatible with OpenAI >= 1.0.0 and Qwen).
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            OpenAI API response
        """
        # Support both old and new OpenAI API
        try:
            # New API (OpenAI >= 1.0.0 and Qwen)
            if hasattr(self.llm_client, 'chat') and hasattr(self.llm_client.chat, 'completions'):
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                return response
            else:
                # Old API (OpenAI < 1.0.0)
                response = self.llm_client.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                return response
        except AttributeError:
            # Fallback to new API structure
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response
    
    def _generate_with_anthropic(self, prompt: str) -> Any:
        """
        Generate text using Anthropic's API.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Anthropic API response
        """
        response = self.llm_client.completions.create(
            model=self.model_name,
            prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
            max_tokens_to_sample=self.max_tokens,
            temperature=self.temperature
        )
        
        return response
    
    def _generate_with_google(self, prompt: str) -> Any:
        """
        Generate text using Google's Gemini API.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Google API response
        """
        try:
            # Configure generation parameters
            generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "top_p": 0.95,
                "top_k": 40
            }
            
            # Initialize model
            logger.info(f"Generating with Google model: {self.model_name}")
            model = self.llm_client.GenerativeModel(model_name=self.model_name)
            
            # Structure prompt content
            prompt_parts = [prompt]
            
            # Generate content
            response = model.generate_content(prompt_parts)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating with Google API: {e}")
            raise
        
    def _generate_with_huggingface(self, prompt: str) -> Any:
        """
        Generate text using HuggingFace's transformers.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            HuggingFace pipeline output
        """
        response = self.llm_client(
            prompt, 
            max_length=len(prompt.split()) + self.max_tokens, 
            temperature=self.temperature
        )
        
        return response
    
    def _extract_response_text(self, llm_response: Any) -> str:
        """
        Extract generated text from LLM response.
        
        Args:
            llm_response: Response from LLM API
            
        Returns:
            Extracted text
        """
        if self.model_provider == "openai":
            try:
                # Support both old and new OpenAI API response formats
                if hasattr(llm_response, 'choices') and llm_response.choices:
                    if hasattr(llm_response.choices[0], 'message'):
                        return llm_response.choices[0].message.content
                    else:
                        return llm_response.choices[0].content
                else:
                    return str(llm_response)
            except (AttributeError, IndexError):
                logger.warning("Could not extract text from OpenAI response using standard format")
                return str(llm_response)
        
        elif self.model_provider == "anthropic":
            try:
                return llm_response.completion
            except AttributeError:
                logger.warning("Could not extract text from Anthropic response using standard format")
                return str(llm_response)
        
        elif self.model_provider == "google":
            # Gemini responses may return a 'candidates' list with 'content.parts'
            try:
                return llm_response.candidates[0].content.parts[0].text
            except Exception:
                return str(llm_response)
            
        elif self.model_provider == "huggingface":
            try:
                # Extract generated text beyond the prompt
                full_text = llm_response[0]["generated_text"]
                return full_text[len(full_text) - self.max_tokens:]
            except (IndexError, KeyError):
                logger.warning("Could not extract text from HuggingFace response using standard format")
                return str(llm_response)
        
        else:
            logger.warning(f"Unknown model provider: {self.model_provider}. Returning string representation.")
            return str(llm_response)
    
    def generate_with_reflexive_correction(self, 
                                         query: str, 
                                         context: Dict[str, Any],
                                         validator: Any = None,
                                         max_attempts: int = 2) -> Dict[str, Any]:
        """
        Generate a response with reflexive contradiction correction.
        
        Args:
            query: User query
            context: Knowledge graph context
            validator: Human validator instance
            max_attempts: Maximum number of correction attempts
            
        Returns:
            Generated response
        """
        # Generate initial response
        response = self.generate(query, context)
        
        # Check for contradictions
        contradictions = response.get("meta", {}).get("contradictions", [])
        attempts = 1
        
        # If we have contradictions and a validator, try to correct them
        while contradictions and validator and attempts < max_attempts:
            logger.info(f"Detected {len(contradictions)} contradictions. Attempting correction.")
            
            # Validate the response
            validated_response = validator.validate_response(
                response, contradictions, context
            )
            
            # If the response was modified, use the corrected version
            if validated_response.get("modified", False):
                response["text"] = validated_response.get("corrected_text", response["text"])
                response["meta"]["validator_notes"] = validated_response.get("notes", "")
                response["meta"]["has_contradictions"] = False
                response["meta"]["contradictions"] = []
                response["meta"]["was_corrected"] = True
                break
            
            # If not modified but accepted, keep the original response
            if validated_response.get("accepted", False):
                response["meta"]["validator_notes"] = validated_response.get("notes", "")
                response["meta"]["has_contradictions"] = False  # Validator accepted despite contradictions
                response["meta"]["was_reviewed"] = True
                break
            
            # If rejected, try to regenerate with more explicit context
            enhanced_context = self._enhance_context_for_contradiction(context, contradictions)
            response = self.generate(query, enhanced_context)
            contradictions = response.get("meta", {}).get("contradictions", [])
            attempts += 1
            
            response["meta"]["regeneration_attempts"] = attempts
        
        return response
    
    def _enhance_context_for_contradiction(self, 
                                         context: Dict[str, Any], 
                                         contradictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhance context to address contradictions.
        
        Args:
            context: Original context
            contradictions: Detected contradictions
            
        Returns:
            Enhanced context
        """
        # Create a copy of the original context
        enhanced_context = {
            **context,
            "enhanced_for_contradictions": True
        }
        
        # Collect facts involved in contradictions
        contradiction_facts = []
        for contradiction in contradictions:
            fact = contradiction.get("conflicting_fact")
            if fact:
                contradiction_facts.append(fact)
        
        # Add emphasis to relevant facts
        if "triples" in enhanced_context:
            # Filter and prioritize triples related to contradictions
            prioritized_triples = []
            regular_triples = []
            
            for triple in enhanced_context["triples"]:
                triple_str = f"{triple.get('subject')} {triple.get('predicate')} {triple.get('object')}"
                
                # Check if this triple relates to contradictions
                is_priority = any(fact in triple_str for fact in contradiction_facts)
                
                if is_priority:
                    # Mark as high priority
                    triple["priority"] = "high"
                    prioritized_triples.append(triple)
                else:
                    regular_triples.append(triple)
            
            # Reorder triples to put prioritized ones first
            enhanced_context["triples"] = prioritized_triples + regular_triples
        
        return enhanced_context