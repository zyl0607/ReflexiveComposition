# reflexive_composition/knowledge_graph/subgraph.py
"""
Subgraph retrieval for knowledge-guided LLM inference.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set, Tuple

logger = logging.getLogger(__name__)

class SubgraphRetriever:
    """
    Retrieves relevant subgraphs from the knowledge graph for LLM context.
    
    This class implements various strategies for identifying and extracting
    subgraphs relevant to a given query, optimizing for context window
    constraints and relevance.
    """
    
    def __init__(self, storage):
        """
        Initialize the subgraph retriever.
        
        Args:
            storage: The graph storage instance
        """
        self.storage = storage
    
    def retrieve_relevant_subgraph(self, 
                                   query: str, 
                                   max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve a subgraph relevant to the query.
        
        Args:
            query: Natural language query
            max_items: Maximum number of triples to include
            
        Returns:
            List of triples forming the relevant subgraph
        """
        # Extract key entities and terms from the query
        entities = self._extract_entities(query)
        
        # Collect candidate triples
        candidate_triples = []
        
        # Get triples for each entity
        for entity in entities:
            entity_data = self.storage.get_entity(entity)
            candidate_triples.extend(entity_data.get("triples", []))
        
        # If no exact matches, try keyword-based retrieval
        if not candidate_triples:
            keywords = self._extract_keywords(query)
            for keyword in keywords:
                results = self.storage.query(keyword, "regex")
                candidate_triples.extend(results)
        
        # Score and rank triples by relevance
        scored_triples = self._score_triples(candidate_triples, query)
        
        # Select top-ranked triples up to max_items
        return [triple for _, triple in scored_triples[:max_items]]
    
    def _extract_entities(self, query: str) -> List[str]:
        """
        Extract potential entity names from a query.
        
        Args:
            query: Natural language query
            
        Returns:
            List of potential entity names
        """
        # Simple heuristic: extract capitalized phrases
        # In a real implementation, we might use NER or more sophisticated methods
        entity_pattern = r'\b[A-Z][a-zA-Z]*\b'
        matches = re.findall(entity_pattern, query)
        
        # Remove common words that might be capitalized
        stopwords = {"The", "A", "An", "In", "On", "At", "By", "For", "With", "About"}
        entities = [entity for entity in matches if entity not in stopwords]
        
        return entities
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract relevant keywords from a query.
        
        Args:
            query: Natural language query
            
        Returns:
            List of keywords
        """
        # Remove common stopwords
        stopwords = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", 
            "by", "for", "with", "about", "is", "are", "was", "were"
        }
        
        # Extract words, convert to lowercase, and filter out stopwords
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stopwords]
        
        return keywords
    
    def _score_triples(self, 
                      triples: List[Dict[str, Any]], 
                      query: str) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Score triples by relevance to the query.
        
        Args:
            triples: List of triple dictionaries
            query: Natural language query
            
        Returns:
            List of (score, triple) tuples, sorted by descending score
        """
        query_terms = set(self._extract_keywords(query))
        scored_triples = []
        
        # Remove duplicates from triples
        unique_triples = []
        seen = set()
        for triple in triples:
            # Create a hashable representation of the triple
            triple_key = (
                triple.get("subject", ""), 
                triple.get("predicate", ""), 
                triple.get("object", "")
            )
            
            if triple_key not in seen:
                seen.add(triple_key)
                unique_triples.append(triple)
        
        # Score each unique triple
        for triple in unique_triples:
            score = self._calculate_triple_score(triple, query_terms)
            scored_triples.append((score, triple))
        
        # Sort by descending score
        return sorted(scored_triples, key=lambda x: x[0], reverse=True)
    
    def _calculate_triple_score(self, 
                               triple: Dict[str, Any], 
                               query_terms: Set[str]) -> float:
        """
        Calculate a relevance score for a triple.
        
        Args:
            triple: Triple dictionary
            query_terms: Set of terms from the query
            
        Returns:
            Relevance score
        """
        # Extract terms from the triple
        triple_terms = set()
        for field in ["subject", "predicate", "object"]:
            value = triple.get(field, "")
            if isinstance(value, str):
                # Extract words from the value
                words = re.findall(r'\b\w+\b', value.lower())
                triple_terms.update(words)
        
        # Calculate term overlap
        overlap = len(query_terms.intersection(triple_terms))
        
        # Base score on overlap
        score = overlap / max(1, len(query_terms))
        
        # Boost score based on additional factors
        
        # Boost for subject match (entity focus)
        subject = triple.get("subject", "").lower()
        if any(term in subject for term in query_terms):
            score += 0.3
        
        # Boost for confidence
        confidence = triple.get("confidence", 0.0)
        score += confidence * 0.2
        
        return score
    
    def graph_expansion(self, 
                        seed_entities: List[str], 
                        max_hops: int = 1, 
                        max_nodes: int = 20) -> List[Dict[str, Any]]:
        """
        Expand a subgraph from seed entities.
        
        Args:
            seed_entities: List of entity IDs to start from
            max_hops: Maximum number of hops for graph traversal
            max_nodes: Maximum number of nodes to include
            
        Returns:
            Expanded subgraph as a list of triples
        """
        visited = set()
        subgraph = []
        frontier = seed_entities.copy()
        
        # BFS graph traversal
        for hop in range(max_hops + 1):
            if not frontier or len(visited) >= max_nodes:
                break
            
            next_frontier = []
            
            for entity in frontier:
                if entity in visited:
                    continue
                
                visited.add(entity)
                
                # Get triples for the entity
                entity_data = self.storage.get_entity(entity)
                entity_triples = entity_data.get("triples", [])
                
                # Add to subgraph
                subgraph.extend(entity_triples)
                
                # Add connected entities to next frontier
                for triple in entity_triples:
                    if triple.get("subject") == entity:
                        next_frontier.append(triple.get("object"))
                    elif triple.get("object") == entity:
                        next_frontier.append(triple.get("subject"))
            
            frontier = next_frontier
        
        return subgraph
    
    def temporal_filter(self, 
                        triples: List[Dict[str, Any]], 
                        reference_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Filter triples based on temporal relevance.
        
        Args:
            triples: List of triple dictionaries
            reference_date: Reference date for filtering
            
        Returns:
            Temporally filtered triples
        """
        if not reference_date:
            return triples
        
        # In a real implementation, we would parse dates and compare them
        # For this example, we'll just look for date strings in the triples
        filtered_triples = []
        
        for triple in triples:
            # Check if this is a timestamp triple
            if triple.get("predicate") in ["timestamp", "date", "createdAt", "updatedAt"]:
                # Compare with reference date (simplified)
                if triple.get("object") <= reference_date:
                    filtered_triples.append(triple)
            else:
                # Non-timestamp triples pass through
                filtered_triples.append(triple)
        
        return filtered_triples