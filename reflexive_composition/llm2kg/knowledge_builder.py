# reflexive_composition/llm2kg/knowledge_builder.py
"""
KnowledgeBuilderLLM: Core component for LLM-based knowledge extraction.

This class handles the interaction with the LLM to extract structured knowledge
from unstructured text, guided by a schema.
"""

import json
import re
import logging
from typing import Dict, List, Optional, Any, Union, Callable

logger = logging.getLogger(__name__)

class KnowledgeBuilderLLM:
    """
    Knowledge Builder LLM that extracts structured knowledge from text.
    
    This class is responsible for prompting the LLM to extract entities,
    relationships, and attributes from text in a format that aligns with
    the target knowledge graph schema.
    """
    
    def __init__(self, 
                 model_name: str,
                 api_key: Optional[str] = None,
                 model_provider: str = "openai",
                 extraction_prompt_template: Optional[str] = None,
                 schema_prompt_template: Optional[str] = None,
                 max_tokens: int = 1000,
                 temperature: float = 0.2,
                 custom_llm_client: Optional[Any] = None):
        """
        Initialize the Knowledge Builder LLM.
        
        Args:
            model_name: Name/identifier of the LLM
            api_key: API key for accessing the LLM service
            model_provider: Provider of the LLM (openai, anthropic, etc.)
            extraction_prompt_template: Template for knowledge extraction prompts
            schema_prompt_template: Template for schema-guided extraction
            max_tokens: Maximum tokens for LLM generation
            temperature: LLM temperature parameter
            custom_llm_client: Optional custom LLM client implementation
        """
        self.model_name = model_name
        self.api_key = api_key
        self.model_provider = model_provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Set default extraction prompt template if none provided
        self.extraction_prompt_template = extraction_prompt_template or """
        Extract structured knowledge from the following text.
        Return the extracted information as JSON with entities, relationships, and attributes.
        
        Text: {text}
        
        {schema_guidance}
        
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
        
        # Set default schema prompt template if none provided
        self.schema_prompt_template = schema_prompt_template or """
        Schema information:
        Entity types: {entity_types}
        Relationship types: {relationship_types}
        
        Make sure all extracted information follows this schema.
        """
        
        # Initialize the LLM client
        self.llm_client = custom_llm_client or self._init_llm_client()
        
        # Initialize extractor helper
        from .extractor import KnowledgeExtractor
        self.extractor = KnowledgeExtractor()
        
        # Initialize schema manager
        from .schema_evolution import SchemaManager
        self.schema_manager = SchemaManager()
    
    def _init_llm_client(self) -> Any:
        """
        Initialize the appropriate LLM client based on the provider.
        
        Returns:
            Initialized LLM client
        """
        if self.model_provider == "openai":
            try:
                import openai
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
    
    def extract(self, 
               text_or_prompt: str, 
               schema: Optional[Dict[str, Any]] = None,
               extraction_type: str = "general",
               domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract structured knowledge from text or a formatted prompt.
        
        Args:
            text_or_prompt: Source text or formatted prompt
            schema: Optional schema to guide extraction
            extraction_type: Type of extraction ("general", "temporal", "domain")
            domain: Domain for domain-specific extraction
            
        Returns:
            Extracted structured knowledge as a dictionary
        """
        # Check if this is already a formatted prompt
        is_formatted_prompt = "Extract structured knowledge" in text_or_prompt or "Extract temporal knowledge" in text_or_prompt
        
        if is_formatted_prompt:
            # Use the provided prompt directly
            prompt = text_or_prompt
        else:
            # Format the prompt based on extraction type
            from .prompt_templates import (
                get_extraction_prompt, 
                get_temporal_extraction_prompt,
                get_domain_specific_extraction_prompt
            )
            
            if extraction_type == "temporal":
                prompt = get_temporal_extraction_prompt(text_or_prompt, schema)
            elif extraction_type == "domain" and domain:
                prompt = get_domain_specific_extraction_prompt(text_or_prompt, domain, schema)
            else:
                # Default to general extraction
                prompt = get_extraction_prompt(text_or_prompt, schema)
        
        # Generate extraction with the appropriate LLM client
        extraction_result = self._generate_with_llm(prompt)
        
        # Parse and validate the extraction
        parsed_extraction = self._parse_extraction(extraction_result)
        
        # Process the extraction for schema compatibility
        if schema and hasattr(self, 'schema_manager'):
            parsed_extraction = self.schema_manager.validate_against_schema(
                parsed_extraction, schema
            )
        
        return parsed_extraction
    
    def _generate_with_llm(self, prompt: str) -> str:
        """
        Generate text using the configured LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Generated text from the LLM
        """
        try:
            if self.model_provider == "openai":
                response = self.llm_client.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                return response.choices[0].message.content
            
            elif self.model_provider == "anthropic":
                response = self.llm_client.completions.create(
                    model=self.model_name,
                    prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
                    max_tokens_to_sample=self.max_tokens,
                    temperature=self.temperature
                )
                return response.completion
            
            elif self.model_provider == "google":  # Add Google Gemini support
                generation_config = {
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "top_p": 0.95,
                    "top_k": 40
                }
                
                model = self.llm_client.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=generation_config
                )
                
                response = model.generate_content(prompt)
                
                # Handle response based on Google's API structure
                if hasattr(response, 'text'):
                    return response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    return response.candidates[0].content.parts[0].text
                else:
                    return str(response)
                
            elif self.model_provider == "huggingface":
                response = self.llm_client(
                    prompt, 
                    max_length=self.max_tokens, 
                    temperature=self.temperature
                )
                return response[0]["generated_text"].replace(prompt, "")
            
            else:
                raise ValueError(f"Unsupported model provider for generation: {self.model_provider}")
        
        except Exception as e:
            logger.error(f"Error generating extraction: {e}")
            return ""
    
    def _parse_extraction(self, extraction_text: str) -> Dict[str, Any]:
        """
        Parse the extraction result from the LLM.
        
        Args:
            extraction_text: Text generated by the LLM
            
        Returns:
            Parsed structured data
        """
        try:
            # Try to find JSON in the response
            start_idx = extraction_text.find("{")
            end_idx = extraction_text.rfind("}")
            
            if start_idx >= 0 and end_idx >= 0:
                json_str = extraction_text[start_idx:end_idx+1]
                extraction_data = json.loads(json_str)
                
                # Ensure it has the expected structure
                if "triples" not in extraction_data:
                    extraction_data = {"triples": extraction_data}
                
                # Ensure each triple has a confidence score if missing
                for triple in extraction_data.get("triples", []):
                    if "confidence" not in triple:
                        triple["confidence"] = 0.8  # Default confidence
                
                return extraction_data
            
            # If JSON parsing fails, use the extractor to try to parse in other formats
            return self.extractor.extract_from_text(extraction_text)
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from extraction, falling back to heuristic parsing")
            logger.warning(f"Extraction text: {extraction_text}")
            truncated_json = self._try_recover_json(extraction_text)
            if truncated_json:
                return truncated_json
            logger.warning("Recovery failed, using extractor for parsing")
            return self.extractor.extract_from_text(extraction_text)
        
        except Exception as e:
            logger.error(f"Error parsing extraction: {e}")
            return {"triples": []}
    
    def _try_recover_json(self, text):
        """Attempt to recover from a truncated JSON array of triples."""
        match = re.search(r"\{.*\"triples\":\s*\[", text, re.DOTALL)
        if not match:
            return None
        start = match.start()
        truncated = text[start:]
        while truncated and not truncated.strip().endswith("}"):
            truncated = truncated.strip()[:-1]
        try:
            return json.loads(truncated)
        except Exception:
            return None
        
    def suggest_schema_updates(self, 
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
        return self.schema_manager.suggest_updates(extractions, current_schema)