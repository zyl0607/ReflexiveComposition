# reflexive_composition/knowledge_graph/storage.py
"""
Storage implementations for Knowledge Graph persistence.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union, Set

logger = logging.getLogger(__name__)

class GraphStorage:
    """
    Knowledge Graph storage layer.
    
    Handles storing, retrieving, and querying triples, with support for
    different backend implementations.
    """
    
    def __init__(self, 
                 storage_type: str = "in_memory",
                 connection_string: Optional[str] = None):
        """
        Initialize the storage backend.
        
        Args:
            storage_type: Type of storage ("in_memory", "rdf", "neo4j", etc.)
            connection_string: Connection string for persistent storage
        """
        self.storage_type = storage_type
        self.connection_string = connection_string
        
        # Initialize the appropriate storage backend
        if storage_type == "in_memory":
            self.backend = InMemoryStorage()
        elif storage_type == "rdf":
            self.backend = RDFStorage(connection_string)
        elif storage_type == "neo4j":
            self.backend = Neo4jStorage(connection_string)
        else:
            logger.warning(f"Unsupported storage type: {storage_type}. Falling back to in-memory storage.")
            self.backend = InMemoryStorage()
    
    def add_triples(self, triples: List[Dict[str, Any]]) -> bool:
        """
        Add triples to storage.
        
        Args:
            triples: List of triple dictionaries
            
        Returns:
            Success status
        """
        return self.backend.add_triples(triples)
    
        
    def get_triples(self) -> List[Dict[str, Any]]:
        """
        Retrieve all triples from the storage backend.

                Returns:
            List of triple dictionaries.
        """
        try:
            return self.backend.triples if hasattr(self.backend, "triples") else []
        except Exception as e:
            logger.error(f"Error retrieving triples from backend: {e}")
            return []
    
    def query(self, query_string: str, query_type: str = "sparql") -> List[Dict[str, Any]]:
        """
        Query the storage.
        
        Args:
            query_string: Query string in the specified format
            query_type: Query language/format
            
        Returns:
            Query results
        """
        return self.backend.query(query_string, query_type)
    
    def get_entity(self, entity_id: str) -> Dict[str, Any]:
        """
        Get all information about an entity.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Entity data
        """
        return self.backend.get_entity(entity_id)
    
    def update_schema(self, schema: Dict[str, Any]) -> bool:
        """
        Update the storage schema.
        
        Args:
            schema: New schema definition
            
        Returns:
            Success status
        """
        return self.backend.update_schema(schema)
    
    def get_triple_count(self) -> int:
        """
        Get the total number of triples.
        
        Returns:
            Triple count
        """
        return self.backend.get_triple_count()
    
    def get_entity_count(self) -> int:
        """
        Get the total number of entities.
        
        Returns:
            Entity count
        """
        return self.backend.get_entity_count()


class InMemoryStorage:
    """
    In-memory implementation of graph storage.
    """
    
    def __init__(self):
        """Initialize in-memory storage."""
        self.triples = []
        self.entities = {}  # entity_id -> entity data
        self.schema = {}
    
    def add_triples(self, triples: List[Dict[str, Any]]) -> bool:
        """
        Add triples to in-memory storage.
        
        Args:
            triples: List of triple dictionaries
            
        Returns:
            Success status
        """
        # Add triples to storage
        self.triples.extend(triples)
        
        # Update entity index
        for triple in triples:
            subject = triple.get("subject")
            obj = triple.get("object")
            
            # Update subject entity
            if subject not in self.entities:
                self.entities[subject] = {"id": subject, "triples": []}
            self.entities[subject]["triples"].append(triple)
            
            # Update object entity if not a literal
            if isinstance(obj, str) and not obj.startswith('"') and not obj.isnumeric():
                if obj not in self.entities:
                    self.entities[obj] = {"id": obj, "triples": []}
                self.entities[obj]["triples"].append(triple)
        
        return True
    
    def query(self, query_string: str, query_type: str = "sparql") -> List[Dict[str, Any]]:
        """
        Query the in-memory storage.
        
        Args:
            query_string: Query string in the specified format
            query_type: Query language/format
            
        Returns:
            Query results
        """
        # For in-memory, implement a simple regex-based query
        # This is a very basic implementation
        results = []
        
        try:
            if query_type == "regex":
                import re
                pattern = re.compile(query_string, re.IGNORECASE)
                
                for triple in self.triples:
                    subject = triple.get("subject", "")
                    predicate = triple.get("predicate", "")
                    obj = triple.get("object", "")
                    
                    if (isinstance(subject, str) and pattern.search(subject)) or \
                       (isinstance(predicate, str) and pattern.search(predicate)) or \
                       (isinstance(obj, str) and pattern.search(obj)):
                        results.append(triple)
            
            elif query_type == "entity_predicate":
                # Format: "entity:predicate" to find all triples with the given entity and predicate
                if ":" in query_string:
                    entity, predicate = query_string.split(":", 1)
                    entity = entity.strip()
                    predicate = predicate.strip()
                    
                    for triple in self.triples:
                        if triple.get("subject") == entity and triple.get("predicate") == predicate:
                            results.append(triple)
            
            else:
                logger.warning(f"Unsupported query type for in-memory storage: {query_type}")
        
        except Exception as e:
            logger.error(f"Error querying in-memory storage: {e}")
        
        return results
    
    def get_entity(self, entity_id: str) -> Dict[str, Any]:
        """
        Get all information about an entity.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Entity data
        """
        return self.entities.get(entity_id, {"id": entity_id, "triples": []})
    
    def update_schema(self, schema: Dict[str, Any]) -> bool:
        """
        Update the storage schema.
        
        Args:
            schema: New schema definition
            
        Returns:
            Success status
        """
        self.schema = schema
        return True
    
    def get_triple_count(self) -> int:
        """
        Get the total number of triples.
        
        Returns:
            Triple count
        """
        return len(self.triples)
    
    def get_entity_count(self) -> int:
        """
        Get the total number of entities.
        
        Returns:
            Entity count
        """
        return len(self.entities)


class RDFStorage:
    """
    RDF-based implementation of graph storage.
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize RDF storage.
        
        Args:
            connection_string: Path to RDF file or endpoint
        """
        self.connection_string = connection_string
        
        try:
            import rdflib
            self.graph = rdflib.Graph()
            
            # Load existing data if connection string is a file path
            if connection_string and connection_string.endswith(('.ttl', '.nt', '.n3', '.xml', '.rdf')):
                try:
                    self.graph.parse(connection_string)
                    logger.info(f"Loaded RDF data from {connection_string}")
                except Exception as e:
                    logger.warning(f"Could not load RDF data from {connection_string}: {e}")
                    
        except ImportError:
            logger.error("RDFLib not installed. Please install with 'pip install rdflib'")
            raise
    
    def add_triples(self, triples: List[Dict[str, Any]]) -> bool:
        """
        Add triples to RDF storage.
        
        Args:
            triples: List of triple dictionaries
            
        Returns:
            Success status
        """
        try:
            import rdflib
            from rdflib import URIRef, Literal, BNode
            
            # Convert dictionary triples to RDF triples
            for triple in triples:
                subject = triple.get("subject")
                predicate = triple.get("predicate")
                obj = triple.get("object")
                
                # Create URIRefs or Literals
                s = URIRef(subject) if "://" in subject else BNode(subject)
                p = URIRef(predicate) if "://" in predicate else URIRef(f"http://example.org/{predicate}")
                
                # If object is a literal, convert appropriately
                if isinstance(obj, str) and (obj.startswith('"') or obj.isnumeric()):
                    o = Literal(obj)
                else:
                    o = URIRef(obj) if "://" in obj else BNode(obj)
                
                # Add to graph
                self.graph.add((s, p, o))
            
            # Save the graph if a file path is provided
            if self.connection_string and self.connection_string.endswith(('.ttl', '.nt', '.n3', '.xml', '.rdf')):
                self.graph.serialize(destination=self.connection_string, format="turtle")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding triples to RDF storage: {e}")
            return False

    def query(self, query_string: str, query_type: str = "sparql") -> List[Dict[str, Any]]:
        """
        Query the RDF storage.
        
        Args:
            query_string: Query string in the specified format
            query_type: Query language/format
            
        Returns:
            Query results
        """
        try:
            if query_type != "sparql":
                logger.warning(f"Unsupported query type for RDF storage: {query_type}. Using SPARQL.")
            
            results = []
            
            # Execute SPARQL query
            qres = self.graph.query(query_string)
            
            # Convert to dictionary format
            for row in qres:
                result = {}
                for i, var in enumerate(qres.vars):
                    result[str(var)] = str(row[i])
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying RDF storage: {e}")
            return []
    
    def get_entity(self, entity_id: str) -> Dict[str, Any]:
        """
        Get all information about an entity in RDF storage.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Entity data
        """
        try:
            import rdflib
            from rdflib import URIRef, BNode
            
            # Create URIRef or BNode for entity
            entity = URIRef(entity_id) if "://" in entity_id else BNode(entity_id)
            
            triples = []
            
            # Get all triples where entity is subject
            for s, p, o in self.graph.triples((entity, None, None)):
                triples.append({
                    "subject": str(s),
                    "predicate": str(p),
                    "object": str(o)
                })
            
            # Get all triples where entity is object
            for s, p, o in self.graph.triples((None, None, entity)):
                triples.append({
                    "subject": str(s),
                    "predicate": str(p),
                    "object": str(o)
                })
            
            return {"id": entity_id, "triples": triples}
            
        except Exception as e:
            logger.error(f"Error getting entity from RDF storage: {e}")
            return {"id": entity_id, "triples": []}
    
    def update_schema(self, schema: Dict[str, Any]) -> bool:
        """
        Update the RDF storage schema.
        
        Args:
            schema: New schema definition
            
        Returns:
            Success status
        """
        # For RDF, we would add schema triples to the graph
        # This is a simplified version
        try:
            import rdflib
            from rdflib import URIRef, Literal, Namespace
            
            # Create schema namespace
            SCHEMA = Namespace("http://schema.org/")
            
            # Clear existing schema triples
            # This is a simplification; in practice, we'd be more selective
            self.graph.remove((URIRef("http://example.org/schema"), None, None))
            
            # Add schema version
            self.graph.add((
                URIRef("http://example.org/schema"),
                SCHEMA.version,
                Literal(schema.get("version", 0))
            ))
            
            # Add entity types
            for entity_type in schema.get("entity_types", []):
                self.graph.add((
                    URIRef("http://example.org/schema"),
                    SCHEMA.defines,
                    URIRef(f"http://example.org/entityType/{entity_type}")
                ))
            
            # Add relationship types
            for relationship_type in schema.get("relationship_types", []):
                self.graph.add((
                    URIRef("http://example.org/schema"),
                    SCHEMA.defines,
                    URIRef(f"http://example.org/relationshipType/{relationship_type}")
                ))
            
            # Save the graph if a file path is provided
            if self.connection_string and self.connection_string.endswith(('.ttl', '.nt', '.n3', '.xml', '.rdf')):
                self.graph.serialize(destination=self.connection_string, format="turtle")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating schema in RDF storage: {e}")
            return False
    
    def get_triple_count(self) -> int:
        """
        Get the total number of triples in RDF storage.
        
        Returns:
            Triple count
        """
        return len(self.graph)
    
    def get_entity_count(self) -> int:
        """
        Get the total number of entities in RDF storage.
        
        Returns:
            Entity count
        """
        try:
            # Get unique subjects and objects
            subjects = set()
            objects = set()
            
            for s, p, o in self.graph:
                subjects.add(str(s))
                objects.add(str(o))
            
            # Combine and count unique entities
            entities = subjects.union(objects)
            return len(entities)
            
        except Exception as e:
            logger.error(f"Error counting entities in RDF storage: {e}")
            return 0


class Neo4jStorage:
    """
    Neo4j-based implementation of graph storage.
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize Neo4j storage.
        
        Args:
            connection_string: Neo4j connection URI with auth
        """
        self.connection_string = connection_string
        
        try:
            from neo4j import GraphDatabase
            
            # Parse connection string
            if connection_string:
                self.driver = GraphDatabase.driver(connection_string)
                self.session = self.driver.session()
                logger.info("Connected to Neo4j database")
            else:
                logger.warning("No Neo4j connection string provided")
                self.driver = None
                self.session = None
                
        except ImportError:
            logger.error("Neo4j driver not installed. Please install with 'pip install neo4j'")
            raise
    
    def add_triples(self, triples: List[Dict[str, Any]]) -> bool:
        """
        Add triples to Neo4j storage.
        
        Args:
            triples: List of triple dictionaries
            
        Returns:
            Success status
        """
        if not self.session:
            logger.error("No Neo4j session available")
            return False
        
        try:
            # For Neo4j, we create nodes and relationships
            for triple in triples:
                subject = triple.get("subject")
                predicate = triple.get("predicate")
                obj = triple.get("object")
                
                # Create query to merge nodes and create relationship
                query = """
                MERGE (s:Entity {id: $subject})
                MERGE (o:Entity {id: $object})
                CREATE (s)-[r:RELATIONSHIP {type: $predicate}]->(o)
                RETURN s, r, o
                """
                
                # Execute query
                self.session.run(query, subject=subject, predicate=predicate, object=obj)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding triples to Neo4j storage: {e}")
            return False
    
    def query(self, query_string: str, query_type: str = "cypher") -> List[Dict[str, Any]]:
        """
        Query the Neo4j storage.
        
        Args:
            query_string: Query string in the specified format
            query_type: Query language/format
            
        Returns:
            Query results
        """
        if not self.session:
            logger.error("No Neo4j session available")
            return []
        
        try:
            if query_type != "cypher":
                logger.warning(f"Unsupported query type for Neo4j storage: {query_type}. Using Cypher.")
            
            # Execute Cypher query
            result = self.session.run(query_string)
            
            # Convert to dictionary format
            records = []
            for record in result:
                records.append(dict(record))
            
            return records
            
        except Exception as e:
            logger.error(f"Error querying Neo4j storage: {e}")
            return []
    
    def get_entity(self, entity_id: str) -> Dict[str, Any]:
        """
        Get all information about an entity in Neo4j storage.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Entity data
        """
        if not self.session:
            logger.error("No Neo4j session available")
            return {"id": entity_id, "triples": []}
        
        try:
            # Query to get all relationships involving the entity
            query = """
            MATCH (s:Entity {id: $entity_id})-[r]->(o)
            RETURN s.id AS subject, type(r) AS predicate, o.id AS object
            UNION
            MATCH (s)-[r]->(o:Entity {id: $entity_id})
            RETURN s.id AS subject, type(r) AS predicate, o.id AS object
            """
            
            result = self.session.run(query, entity_id=entity_id)
            
            triples = []
            for record in result:
                triples.append({
                    "subject": record["subject"],
                    "predicate": record["predicate"],
                    "object": record["object"]
                })
            
            return {"id": entity_id, "triples": triples}
            
        except Exception as e:
            logger.error(f"Error getting entity from Neo4j storage: {e}")
            return {"id": entity_id, "triples": []}
    
    def update_schema(self, schema: Dict[str, Any]) -> bool:
        """
        Update the Neo4j storage schema.
        
        Args:
            schema: New schema definition
            
        Returns:
            Success status
        """
        if not self.session:
            logger.error("No Neo4j session available")
            return False
        
        try:
            # For Neo4j, we would create schema constraints and indexes
            # This is a simplified version
            
            # Create schema node
            query = """
            MERGE (s:Schema {id: 'schema'})
            SET s.version = $version
            RETURN s
            """
            
            self.session.run(query, version=schema.get("version", 0))
            
            # Create entity type nodes
            for entity_type in schema.get("entity_types", []):
                query = """
                MERGE (s:Schema {id: 'schema'})
                MERGE (t:EntityType {name: $entity_type})
                MERGE (s)-[:DEFINES]->(t)
                RETURN t
                """
                
                self.session.run(query, entity_type=entity_type)
            
            # Create relationship type nodes
            for relationship_type in schema.get("relationship_types", []):
                query = """
                MERGE (s:Schema {id: 'schema'})
                MERGE (r:RelationshipType {name: $relationship_type})
                MERGE (s)-[:DEFINES]->(r)
                RETURN r
                """
                
                self.session.run(query, relationship_type=relationship_type)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating schema in Neo4j storage: {e}")
            return False
    
    def get_triple_count(self) -> int:
        """
        Get the total number of triples (relationships) in Neo4j storage.
        
        Returns:
            Triple count
        """
        if not self.session:
            logger.error("No Neo4j session available")
            return 0
        
        try:
            query = """
            MATCH ()-[r:RELATIONSHIP]->()
            RETURN count(r) AS count
            """
            
            result = self.session.run(query)
            record = result.single()
            
            return record["count"] if record else 0
            
        except Exception as e:
            logger.error(f"Error counting triples in Neo4j storage: {e}")
            return 0
    
    def get_entity_count(self) -> int:
        """
        Get the total number of entities in Neo4j storage.
        
        Returns:
            Entity count
        """
        if not self.session:
            logger.error("No Neo4j session available")
            return 0
        
        try:
            query = """
            MATCH (n:Entity)
            RETURN count(n) AS count
            """
            
            result = self.session.run(query)
            record = result.single()
            
            return record["count"] if record else 0
            
        except Exception as e:
            logger.error(f"Error counting entities in Neo4j storage: {e}")
            return 0
    
    def __del__(self):
        """Close Neo4j connections on object destruction."""
        if hasattr(self, 'session') and self.session:
            self.session.close()
        
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()