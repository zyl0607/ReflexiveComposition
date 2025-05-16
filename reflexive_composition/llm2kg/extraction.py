# reflexive_composition/llm2kg/extraction.py
"""
Knowledge extraction utilities for parsing structured data from LLM outputs.
"""

import re
import json
import logging
from typing import Dict, List, Any, Union, Optional

logger = logging.getLogger(__name__)

class KnowledgeExtractor:
    """
    Extracts structured knowledge from LLM-generated text.
    
    This class handles parsing and structuring of knowledge from various text
    formats including JSON, triples, key-value pairs, and tabular data.
    """
    
    def __init__(self):
        """Initialize the knowledge extractor."""
        pass
    
    def extract_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract structured knowledge from text using multiple extraction methods.
        
        Args:
            text: Text to extract knowledge from
            
        Returns:
            Extracted knowledge in a standardized format
        """
        # Try different extraction methods in order of preference
        extractors = [
            self._extract_json,
            self._extract_triples,
            self._extract_key_value_pairs,
            self._extract_tabular
        ]
        
        for extractor in extractors:
            result = extractor(text)
            if result and result.get("triples") and len(result.get("triples")) > 0:
                return result
        
        # If all extractors fail, return empty result
        logger.warning("All extraction methods failed. Returning empty result.")
        return {"triples": []}
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON data from text.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Extracted structured data
        """
        try:
            # Find JSON-like patterns
            json_pattern = r'(\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\})'
            matches = re.findall(json_pattern, text)
            
            for match in matches:
                try:
                    data = json.loads(match)
                    
                    # Convert to standard triple format if needed
                    if "triples" not in data:
                        # Check for other common JSON structures
                        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                            # List of objects - treat each as a triple
                            triples = []
                            for item in data:
                                if "subject" in item and "predicate" in item and "object" in item:
                                    triples.append(item)
                                # Handle other common formats like entity-relation structures
                                elif "entity" in item and "relation" in item:
                                    triples.append({
                                        "subject": item.get("entity"),
                                        "predicate": item.get("relation"),
                                        "object": item.get("value", ""),
                                        "confidence": item.get("confidence", 0.8)
                                    })
                            return {"triples": triples}
                        else:
                            # Convert flat key-value pairs to subject-predicate-object
                            entity = data.get("entity", "unknown_entity")
                            triples = []
                            for key, value in data.items():
                                if key != "entity" and key != "triples":
                                    triples.append({
                                        "subject": entity,
                                        "predicate": key,
                                        "object": value,
                                        "confidence": 0.8
                                    })
                            return {"triples": triples}
                    
                    return data
                
                except json.JSONDecodeError:
                    continue
            
            return {"triples": []}
            
        except Exception as e:
            logger.error(f"Error in JSON extraction: {e}")
            return {"triples": []}
    
    def _extract_triples(self, text: str) -> Dict[str, Any]:
        """
        Extract subject-predicate-object triples from text.
        
        Args:
            text: Text containing triple-like statements
            
        Returns:
            Extracted triples
        """
        triples = []
        
        # Pattern for (subject, predicate, object) or similar formats
        triple_patterns = [
            r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)',  # (subject, predicate, object)
            r'([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)',  # subject | predicate | object
            r'([^\t]+)\t([^\t]+)\t([^\t\n]+)'  # subject\tpredicate\tobject
        ]
        
        for pattern in triple_patterns:
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    subject, predicate, obj = [item.strip() for item in match]
                    triples.append({
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "confidence": 0.8  # Default confidence
                    })
        
        # Also look for "Subject: X, Predicate: Y, Object: Z" pattern
        for line in text.split('\n'):
            if 'subject' in line.lower() and 'predicate' in line.lower() and 'object' in line.lower():
                parts = {}
                for part in line.split(','):
                    if ':' in part:
                        key, value = part.split(':', 1)
                        parts[key.strip().lower()] = value.strip()
                
                if 'subject' in parts and 'predicate' in parts and 'object' in parts:
                    triples.append({
                        "subject": parts['subject'],
                        "predicate": parts['predicate'],
                        "object": parts['object'],
                        "confidence": 0.8
                    })
        
        return {"triples": triples}
    
    def _extract_key_value_pairs(self, text: str) -> Dict[str, Any]:
        """
        Extract key-value pairs and convert to triples.
        
        Args:
            text: Text containing key-value pairs
            
        Returns:
            Extracted triples
        """
        triples = []
        entity = "unknown_entity"
        
        # First, try to identify the main entity
        entity_patterns = [
            r'Entity:\s*([^\n]+)',
            r'Name:\s*([^\n]+)',
            r'Title:\s*([^\n]+)',
            r'Subject:\s*([^\n]+)',
            r'About:\s*([^\n]+)'
        ]
        
        for pattern in entity_patterns:
            match = re.search(pattern, text)
            if match:
                entity = match.group(1).strip()
                break
        
        # Extract key-value pairs
        kv_pattern = r'([^:\n]+):\s*([^\n]+)'
        matches = re.findall(kv_pattern, text)
        
        for key, value in matches:
            key = key.strip()
            value = value.strip()
            
            # Skip if this is the entity definition
            if key.lower() in ['entity', 'name', 'title', 'subject', 'about'] and value == entity:
                continue
                
            triples.append({
                "subject": entity,
                "predicate": key,
                "object": value,
                "confidence": 0.8
            })
        
        return {"triples": triples}
    
    def _extract_tabular(self, text: str) -> Dict[str, Any]:
        """
        Extract data from tabular formats.
        
        Args:
            text: Text containing tabular data
            
        Returns:
            Extracted triples
        """
        triples = []
        lines = text.strip().split('\n')
        
        # Check if we have a header line
        if len(lines) >= 2:
            headers = [h.strip() for h in re.split(r'\s{2,}|\t|,', lines[0])]
            
            # Check if we have common triple headers
            if len(headers) >= 3:
                subject_idx = -1
                predicate_idx = -1
                object_idx = -1
                confidence_idx = -1
                
                for i, header in enumerate(headers):
                    header_lower = header.lower()
                    if header_lower in ['subject', 'entity', 'source']:
                        subject_idx = i
                    elif header_lower in ['predicate', 'relation', 'property']:
                        predicate_idx = i
                    elif header_lower in ['object', 'value', 'target']:
                        object_idx = i
                    elif header_lower in ['confidence', 'score', 'probability']:
                        confidence_idx = i
                
                # If we have at least subject, predicate, and object columns
                if subject_idx >= 0 and predicate_idx >= 0 and object_idx >= 0:
                    for i in range(1, len(lines)):
                        cells = [c.strip() for c in re.split(r'\s{2,}|\t|,', lines[i])]
                        
                        if len(cells) >= max(subject_idx, predicate_idx, object_idx) + 1:
                            triple = {
                                "subject": cells[subject_idx],
                                "predicate": cells[predicate_idx],
                                "object": cells[object_idx],
                                "confidence": float(cells[confidence_idx]) if confidence_idx >= 0 and confidence_idx < len(cells) else 0.8
                            }
                            triples.append(triple)
        
        return {"triples": triples}