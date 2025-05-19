# reflexive_composition/kg2llm/contradiction_detector.py
"""
Contradiction detection for knowledge graph enhanced LLM inference.

This module provides methods for detecting contradictions between 
LLM-generated responses and knowledge graph context.
"""

import logging
import re
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Contradiction:
    """Represents a detected contradiction."""
    statement: str
    conflicting_fact: str
    confidence: float
    triple_id: Optional[str] = None
    explanation: Optional[str] = None


class ContradictionDetector:
    """
    Detects contradictions between LLM-generated text and knowledge graph facts.
    """
    
    def __init__(self, 
                 nlp_model: Optional[Any] = None,
                 contradiction_threshold: float = 0.7,
                 use_external_nli: bool = False):
        """
        Initialize the contradiction detector.
        
        Args:
            nlp_model: Optional NLP model for text analysis
            contradiction_threshold: Confidence threshold for contradiction detection
            use_external_nli: Whether to use external NLI service
        """
        self.nlp_model = nlp_model
        self.contradiction_threshold = contradiction_threshold
        self.use_external_nli = use_external_nli
        
        # Initialize NLI model if available
        self.nli_model = None
        if use_external_nli:
            try:
                # Try to import a lightweight NLI model
                # This is optional and only used if available
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                
                model_name = "cross-encoder/nli-deberta-v3-base"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                logger.info(f"Initialized NLI model: {model_name}")
            except (ImportError, Exception) as e:
                logger.warning(f"Failed to initialize NLI model: {e}")
                self.use_external_nli = False
    
    def detect_contradictions(self, 
                             text: str, 
                             context: Any) -> List[Dict[str, Any]]:
        """
        Detect contradictions between generated text and knowledge context.
        
        Args:
            text: Generated text
            context: Knowledge graph context (triples, entities, etc.)
            
        Returns:
            List of detected contradictions
        """
        # Extract statements from the text
        statements = self._extract_statements(text)
        
        # Extract facts from the context
        if isinstance(context, dict) and "triples" in context:
            facts = context["triples"]
        elif isinstance(context, list):
            facts = context
        else:
            facts = []
        
        # Check for contradictions
        contradictions = []
        
        for statement in statements:
            for fact in facts:
                conflict = self._check_contradiction(statement, fact)
                if conflict:
                    contradictions.append({
                        "statement": statement,
                        "conflicting_fact": f"{fact.get('subject', '')} {fact.get('predicate', '')} {fact.get('object', '')}",
                        "triple_id": fact.get("id", None),
                        "confidence": conflict[0],
                        "explanation": conflict[1]
                    })
        
        # Filter and sort contradictions by confidence
        filtered_contradictions = [
            c for c in contradictions 
            if c["confidence"] >= self.contradiction_threshold
        ]
        
        sorted_contradictions = sorted(
            filtered_contradictions, 
            key=lambda x: x["confidence"], 
            reverse=True
        )
        
        return sorted_contradictions
    
    def _extract_statements(self, text: str) -> List[str]:
        """
        Extract factual statements from generated text.
        
        Args:
            text: Generated text
            
        Returns:
            List of extracted statements
        """
        # Simple sentence-based extraction
        # In a real implementation, this would use more sophisticated NLP
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter for likely factual statements
        factual_statements = []
        
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence.split()) < 3:
                continue
                
            # Skip questions
            if sentence.endswith('?'):
                continue
                
            # Skip subjective statements
            subjective_patterns = [
                r'I think', r'I believe', r'In my opinion', 
                r'probably', r'might', r'may', r'could',
                r'seems', r'appears'
            ]
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in subjective_patterns):
                continue
            
            factual_statements.append(sentence)
        
        return factual_statements
    
    def _check_contradiction(self, 
                           statement: str, 
                           fact: Dict[str, Any]) -> Optional[Tuple[float, str]]:
        """
        Check if a statement contradicts a knowledge graph fact.
        
        Args:
            statement: Statement from generated text
            fact: Fact from knowledge graph
            
        Returns:
            Tuple of (confidence, explanation) if contradiction, None otherwise
        """
        # Convert fact to natural language form
        fact_text = f"{fact.get('subject', '')} {fact.get('predicate', '')} {fact.get('object', '')}"
        
        # If NLI model is available, use it
        if self.use_external_nli and self.nli_model:
            return self._check_contradiction_with_nli(statement, fact_text, fact)
        
        # Otherwise, use rule-based detection
        return self._check_contradiction_rule_based(statement, fact)
    
    def _check_contradiction_with_nli(self, 
                                    statement: str, 
                                    fact_text: str,
                                    fact: Dict[str, Any]) -> Optional[Tuple[float, str]]:
        """
        Check for contradiction using Natural Language Inference model.
        
        Args:
            statement: Statement from generated text
            fact_text: Fact text
            fact: Original fact dictionary
            
        Returns:
            Tuple of (confidence, explanation) if contradiction, None otherwise
        """
        try:
            # Tokenize inputs
            inputs = self.tokenizer(
                fact_text, 
                statement, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            )
            
            # Get prediction
            import torch
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1)
            
            # Get contradiction score
            # Typical NLI labels: [entailment, neutral, contradiction]
            contradiction_idx = 2
            contradiction_score = predictions[0, contradiction_idx].item()
            
            if contradiction_score > self.contradiction_threshold:
                return (
                    contradiction_score,
                    f"NLI model detected contradiction between statement and fact with confidence {contradiction_score:.2f}"
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"Error in NLI contradiction detection: {e}")
            # Fall back to rule-based method
            return self._check_contradiction_rule_based(statement, fact)
    
    def _check_contradiction_rule_based(self, 
                                      statement: str, 
                                      fact: Dict[str, Any]) -> Optional[Tuple[float, str]]:
        """
        Check for contradiction using rule-based methods.
        
        Args:
            statement: Statement from generated text
            fact: Fact from knowledge graph
            
        Returns:
            Tuple of (confidence, explanation) if contradiction, None otherwise
        """
        # Extract key elements from fact
        subject = fact.get('subject', '').lower()
        predicate = fact.get('predicate', '').lower()
        obj = fact.get('object', '').lower()
        
        # Skip facts with low confidence
        if float(fact.get('confidence', 1.0)) < 0.5:
            return None
        
        # Check for negation contradictions
        statement_lower = statement.lower()
        
        # Simple negation patterns
        negation_prefixes = ['not ', 'never ', 'no ', 'doesn\'t ', 'don\'t ', 'didn\'t ', 'isn\'t ', 'aren\'t ', 'wasn\'t ', 'weren\'t ']
        
        # Format the fact for pattern matching
        fact_pattern = f"{subject} {predicate} {obj}"
        negated_fact_pattern = None
        
        for prefix in negation_prefixes:
            if f"{subject} {prefix}{predicate} {obj}" in statement_lower:
                # Direct negation found
                return (0.85, f"Statement contains negated form of the fact")
            
            # Also check for negation in middle (e.g., "X is not Y")
            if f"{subject} is {prefix}" in statement_lower and obj in statement_lower:
                return (0.80, f"Statement contains negated relationship with subject and object")
        
        # Check for contradictory values
        # E.g., if fact is "X height 180cm" and statement says "X height 190cm"
        if all(term in statement_lower for term in [subject, predicate]):
            # Subject and predicate are mentioned, but different object value
            if obj not in statement_lower:
                # Extract potential contradictory value
                after_predicate = statement_lower.split(predicate, 1)[1].strip()
                words = after_predicate.split()
                
                # Check if we find a different value
                if len(words) > 0:
                    potential_value = words[0]
                    if potential_value != obj and re.match(r'\w+', potential_value):
                        return (0.75, f"Statement uses different value ({potential_value}) than fact ({obj})")
        
        # Check for temporal contradictions
        if predicate in ['occurred_on', 'happened_on', 'date', 'time', 'on', 'at']:
            # Extract dates from statement
            date_patterns = [
                r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
                r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY or DD/MM/YYYY
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'  # Month DD, YYYY
            ]
            
            dates_in_statement = []
            for pattern in date_patterns:
                dates_in_statement.extend(re.findall(pattern, statement, re.IGNORECASE))
            
            # If statement has a date and it's different from the fact
            if dates_in_statement and not any(obj in date for date in dates_in_statement):
                return (0.80, f"Statement mentions different date ({dates_in_statement[0]}) than fact ({obj})")
        
        # No contradiction detected
        return None
    
    def check_response_factuality(self, 
                                 response: str, 
                                 facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform comprehensive factuality checking on a response.
        
        Args:
            response: Generated response text
            facts: Knowledge graph facts used as context
            
        Returns:
            Factuality assessment
        """
        # Detect contradictions
        contradictions = self.detect_contradictions(response, facts)
        
        # Calculate overall factuality score
        if contradictions:
            # More contradictions = lower factuality
            factuality_score = max(0.0, 1.0 - (sum(c["confidence"] for c in contradictions) / len(contradictions)) * 0.5)
        else:
            factuality_score = 1.0
        
        # Assess grounding - how many facts are actually reflected in the response
        grounding_score = self._assess_grounding(response, facts)
        
        return {
            "factuality_score": factuality_score,
            "grounding_score": grounding_score,
            "contradictions": contradictions,
            "contradiction_count": len(contradictions)
        }
    
    def _assess_grounding(self, 
                        response: str, 
                        facts: List[Dict[str, Any]]) -> float:
        """
        Assess how well the response is grounded in the provided facts.
        
        Args:
            response: Generated response text
            facts: Knowledge graph facts used as context
            
        Returns:
            Grounding score (0.0 to 1.0)
        """
        fact_mentions = 0
        response_lower = response.lower()
        
        for fact in facts:
            subject = fact.get('subject', '').lower()
            predicate = fact.get('predicate', '').lower()
            obj = fact.get('object', '').lower()
            
            # Check if all parts of the fact are mentioned
            if subject in response_lower and obj in response_lower:
                fact_mentions += 1
            
            # Check for naturalized mentions
            # E.g., "X happened on Y" might be expressed as "On Y, X occurred"
            if predicate in ['occurred_on', 'happened_on', 'on', 'at'] and subject in response_lower and obj in response_lower:
                for pattern in [f"on {obj}", f"in {obj}", f"at {obj}", f"{obj},", f"{obj} when"]:
                    if pattern in response_lower:
                        fact_mentions += 0.5
                        break
        
        # Calculate grounding score
        if not facts:
            return 0.0
        
        return min(1.0, fact_mentions / len(facts))
