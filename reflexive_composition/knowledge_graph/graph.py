# reflexive_composition/knowledge_graph/graph.py
"""
Main Knowledge Graph implementation for Reflexive Composition.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Set

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """
    Core Knowledge Graph implementation.
    
    This class handles storage, querying, and updating of the knowledge graph.
    It supports both in-memory and persistent storage options.
    """
    
    def __init__(self, 
                 storage_type: str = "in_memory",
                 connection_string: Optional[str] = None,
                 schema: Optional[Dict[str, Any]] = None):
        """
        Initialize the Knowledge Graph.
        
        Args:
            storage_type: Type of storage ("in_memory", "rdf", "neo4j", etc.)
            connection_string: Connection string for persistent storage
            schema: Initial schema definition
        """
        self.storage_type = storage_type
        self.connection_string = connection_string
        self.schema = schema or {
            "entity_types": [],
            "relationship_types": [],
            "version": 0
        }
        
        # Initialize storage
        from .storage import GraphStorage
        self.storage = GraphStorage(storage_type, connection_string)
        
        # Initialize subgraph retriever
        from .subgraph import SubgraphRetriever
        self.retriever = SubgraphRetriever(self.storage)
        
        # Statistics
        self.stats = {
            "triple_count": 0,
            "entity_count": 0,
            "last_updated": None
        }
    
    def add_triples(self, triples: List[Dict[str, Any]]) -> bool:
        """
        Add triples to the knowledge graph.
        
        Args:
            triples: List of triple dictionaries
            
        Returns:
            Success status
        """
        try:
            # Process and add triples to storage
            self.storage.add_triples(triples)
            
            # Update statistics
            self.stats["triple_count"] += len(triples)
            
            # Update entity count
            entities = set()
            for triple in triples:
                entities.add(triple.get("subject"))
                obj = triple.get("object")
                # Only count objects that are entities (not literals)
                if isinstance(obj, str) and not obj.startswith('"') and not obj.isnumeric():
                    entities.add(obj)
            
            # Update entity count (approximate - storage would have exact count)
            self.stats["entity_count"] = self.storage.get_entity_count()
            
            # Update timestamp
            self.stats["last_updated"] = self._get_timestamp()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding triples to knowledge graph: {e}")
            return False
    
    def get_triples(self) -> List[Dict[str, Any]]:
        """
        Retrieve all triples from the knowledge graph storage.

        Returns:
            List of triples (each as a dict with subject, predicate, object, etc.)
        """
        try:
            return self.storage.get_triples()
        except Exception as e:
            logger.error(f"Error retrieving triples: {e}")
            return []
    
    def query(self, 
              query_string: str, 
              query_type: str = "sparql") -> List[Dict[str, Any]]:
        """
        Query the knowledge graph.
        
        Args:
            query_string: Query string in the specified format
            query_type: Query language/format
            
        Returns:
            Query results
        """
        try:
            return self.storage.query(query_string, query_type)
        except Exception as e:
            logger.error(f"Error querying knowledge graph: {e}")
            return []
    
    def retrieve_context(self, 
                        query: str, 
                        max_items: int = 10) -> Dict[str, Any]:
        """
        Retrieve relevant subgraph for a query.
        
        Args:
            query: Natural language query
            max_items: Maximum number of triples to retrieve
            
        Returns:
            Context dictionary with relevant knowledge
        """
        try:
            subgraph = self.retriever.retrieve_relevant_subgraph(query, max_items)
            return {
                "triples": subgraph,
                "schema": self.schema,
                "query": query
            }
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return {"triples": [], "schema": self.schema, "query": query}
    
    def get_entity(self, entity_id: str) -> Dict[str, Any]:
        """
        Get all information about an entity.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Entity data
        """
        try:
            return self.storage.get_entity(entity_id)
        except Exception as e:
            logger.error(f"Error getting entity: {e}")
            return {}
    
    def update_schema(self, schema: Dict[str, Any]) -> bool:
        """
        Update the knowledge graph schema.
        
        Args:
            schema: New schema definition
            
        Returns:
            Success status
        """
        try:
            old_schema = self.schema.copy()
            self.schema = schema
            self.storage.update_schema(schema)
            
            logger.info(f"Schema updated from version {old_schema.get('version', 0)} to {schema.get('version', 0)}")
            return True
        except Exception as e:
            logger.error(f"Error updating schema: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Knowledge graph statistics
        """
        # Refresh stats from storage
        self.stats["triple_count"] = self.storage.get_triple_count()
        self.stats["entity_count"] = self.storage.get_entity_count()
        
        return self.stats
    
    def _get_timestamp(self) -> str:
        """
        Get current timestamp in ISO format.
        
        Returns:
            Current timestamp string
        """
        from datetime import datetime
        return datetime.utcnow().isoformat()