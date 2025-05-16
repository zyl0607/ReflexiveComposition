import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set

import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecurityKnowledgeGraphBuilder:
    """
    Builds a security-focused knowledge graph for code generation grounding.
    
    This class handles the extraction and structuring of security knowledge from 
    authoritative sources, including API deprecation notices, vulnerability 
    databases, and secure coding guidelines.
    """
    
    def __init__(self, 
                 schema_path: str,
                 kb_llm_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Security Knowledge Graph Builder.
        
        Args:
            schema_path: Path to the security ontology schema
            kb_llm_config: Configuration for the KB-LLM (optional)
        """
        self.schema = self._load_schema(schema_path)
        self.graph = nx.DiGraph()
        
        # Initialize KB-LLM if configuration is provided
        self.kb_llm = None
        if kb_llm_config:
            self._init_kb_llm(kb_llm_config)
            
        # Statistics
        self.stats = {
            "entity_count": 0,
            "relationship_count": 0,
            "source_count": 0,
            "validation_count": 0
        }
            
    def _load_schema(self, schema_path: str) -> Dict[str, Any]:
        """
        Load the security ontology schema.
        
        Args:
            schema_path: Path to the schema file
            
        Returns:
            Loaded schema as dictionary
        """
        try:
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            logger.info(f"Loaded schema with {len(schema.get('entity_types', []))} entity types")
            return schema
        except Exception as e:
            logger.error(f"Error loading schema: {e}")
            return {"entity_types": [], "relationship_types": []}
            
    def _init_kb_llm(self, config: Dict[str, Any]) -> None:
        """
        Initialize the Knowledge Builder LLM.
        
        Args:
            config: KB-LLM configuration
        """
        try:
            model_name = config.get("model_name", "microsoft/codebert-base")
            
            # Initialize tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Create pipeline
            self.kb_llm = pipeline("text-generation", 
                                  model=model, 
                                  tokenizer=tokenizer, 
                                  max_length=config.get("max_length", 512),
                                  temperature=config.get("temperature", 0.3))
            
            logger.info(f"Initialized KB-LLM using {model_name}")
        except Exception as e:
            logger.error(f"Error initializing KB-LLM: {e}")
            self.kb_llm = None
    
    def extract_from_api_docs(self, docs_path: str) -> List[Dict[str, Any]]:
        """
        Extract API deprecation information from documentation.
        
        Args:
            docs_path: Path to the API documentation
            
        Returns:
            List of extracted deprecation triples
        """
        extracted_triples = []
        
        try:
            with open(docs_path, 'r') as f:
                content = f.read()
                
            # Extract deprecated APIs using regex patterns
            # This is a simplified implementation; in practice, more sophisticated 
            # extraction techniques would be used
            deprecation_pattern = r'(?P<api>\w+[\.\w+]*)\s+is\s+deprecated\s+(?:since|in)\s+(?P<version>[\d\.]+)'
            replacement_pattern = r'(?P<old_api>\w+[\.\w+]*)\s+is\s+replaced\s+by\s+(?P<new_api>\w+[\.\w+]*)'
            
            # Find deprecations
            for match in re.finditer(deprecation_pattern, content):
                api_name = match.group('api')
                version = match.group('version')
                
                triple = {
                    "subject": api_name,
                    "predicate": "deprecatedInVersion",
                    "object": version,
                    "confidence": 0.9,
                    "source": docs_path
                }
                extracted_triples.append(triple)
            
            # Find replacements
            for match in re.finditer(replacement_pattern, content):
                old_api = match.group('old_api')
                new_api = match.group('new_api')
                
                triple = {
                    "subject": old_api,
                    "predicate": "replacedBy",
                    "object": new_api,
                    "confidence": 0.9,
                    "source": docs_path
                }
                extracted_triples.append(triple)
                
            logger.info(f"Extracted {len(extracted_triples)} triples from API docs")
            return extracted_triples
        
        except Exception as e:
            logger.error(f"Error extracting from API docs: {e}")
            return []
    
    def extract_from_vulnerability_db(self, vuln_db_path: str) -> List[Dict[str, Any]]:
        """
        Extract vulnerability information from a security database.
        
        Args:
            vuln_db_path: Path to the vulnerability database
            
        Returns:
            List of extracted vulnerability triples
        """
        extracted_triples = []
        
        try:
            with open(vuln_db_path, 'r') as f:
                vuln_db = json.load(f)
            
            for vuln in vuln_db.get('vulnerabilities', []):
                # Create entity for the vulnerability
                vuln_id = vuln.get('id')
                api_name = vuln.get('affected_api')
                risk_level = vuln.get('risk_level', 'Medium')
                
                if api_name and vuln_id:
                    # API has vulnerability
                    triple1 = {
                        "subject": api_name,
                        "predicate": "hasVulnerability",
                        "object": vuln_id,
                        "confidence": 0.95,
                        "source": vuln_db_path
                    }
                    extracted_triples.append(triple1)
                    
                    # Vulnerability risk level
                    triple2 = {
                        "subject": vuln_id,
                        "predicate": "hasRiskLevel",
                        "object": risk_level,
                        "confidence": 0.95,
                        "source": vuln_db_path
                    }
                    extracted_triples.append(triple2)
                    
                    # If secure alternative is provided
                    if secure_alt := vuln.get('secure_alternative'):
                        triple3 = {
                            "subject": api_name,
                            "predicate": "hasSecureAlternative",
                            "object": secure_alt,
                            "confidence": 0.9,
                            "source": vuln_db_path
                        }
                        extracted_triples.append(triple3)
            
            logger.info(f"Extracted {len(extracted_triples)} triples from vulnerability DB")
            return extracted_triples
            
        except Exception as e:
            logger.error(f"Error extracting from vulnerability DB: {e}")
            return []
    
    def extract_from_coding_guidelines(self, guidelines_path: str) -> List[Dict[str, Any]]:
        """
        Extract secure coding patterns from guidelines.
        
        Args:
            guidelines_path: Path to the coding guidelines
            
        Returns:
            List of extracted pattern triples
        """
        # If KB-LLM is available, use it for guideline extraction
        if self.kb_llm:
            return self._extract_with_kb_llm(guidelines_path)
        
        # Fallback to pattern-based extraction
        extracted_triples = []
        
        try:
            with open(guidelines_path, 'r') as f:
                content = f.read()
            
            # Extract patterns using regex
            pattern_sections = re.findall(r'Pattern\s+ID:\s+(\w+).*?Description:\s+(.*?)(?=Pattern\s+ID:|$)', 
                                         content, re.DOTALL)
            
            for pattern_id, description in pattern_sections:
                # Create entity for the pattern
                triple1 = {
                    "subject": pattern_id,
                    "predicate": "type",
                    "object": "SecureCodingPattern",
                    "confidence": 0.9,
                    "source": guidelines_path
                }
                extracted_triples.append(triple1)
                
                # Add description
                triple2 = {
                    "subject": pattern_id,
                    "predicate": "hasDescription",
                    "object": description.strip(),
                    "confidence": 0.85,
                    "source": guidelines_path
                }
                extracted_triples.append(triple2)
                
                # Look for affected APIs in the description
                api_mentions = re.findall(r'`([a-zA-Z0-9_\.]+)`', description)
                for api in api_mentions:
                    triple3 = {
                        "subject": pattern_id,
                        "predicate": "appliesToAPI",
                        "object": api,
                        "confidence": 0.7,  # Lower confidence since this is inferred
                        "source": guidelines_path
                    }
                    extracted_triples.append(triple3)
            
            logger.info(f"Extracted {len(extracted_triples)} triples from coding guidelines")
            return extracted_triples
            
        except Exception as e:
            logger.error(f"Error extracting from coding guidelines: {e}")
            return []
    
    def _extract_with_kb_llm(self, document_path: str) -> List[Dict[str, Any]]:
        """
        Use KB-LLM to extract structured knowledge from a document.
        
        Args:
            document_path: Path to the document
            
        Returns:
            List of extracted triples
        """
        extracted_triples = []
        
        try:
            with open(document_path, 'r') as f:
                content = f.read()
            
            # Prepare prompt for KB-LLM
            entity_types = ", ".join(self.schema.get("entity_types", []))
            relation_types = ", ".join(self.schema.get("relationship_types", []))
            
            prompt = f"""
            Extract structured knowledge from the following security documentation.
            
            Entity types: {entity_types}
            Relationship types: {relation_types}
            
            Return the extraction as a list of triples in JSON format, where each triple has:
            - subject: The entity that the relationship starts from
            - predicate: The type of relationship
            - object: The entity or value that the relationship points to
            
            Document content:
            {content[:4000]}  # Limit content length to avoid token overflow
            
            JSON output:
            """
            
            # Generate extraction using KB-LLM
            result = self.kb_llm(prompt, max_length=1024)[0]["generated_text"]
            
            # Extract JSON from the result
            json_match = re.search(r'\[\s*\{.*\}\s*\]', result, re.DOTALL)
            if json_match:
                try:
                    triples_data = json.loads(json_match.group(0))
                    for triple in triples_data:
                        if all(k in triple for k in ["subject", "predicate", "object"]):
                            # Add confidence and source
                            triple["confidence"] = 0.8  # Standard confidence for LLM extraction
                            triple["source"] = document_path
                            extracted_triples.append(triple)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON from KB-LLM output")
            
            logger.info(f"KB-LLM extracted {len(extracted_triples)} triples from document")
            return extracted_triples
            
        except Exception as e:
            logger.error(f"Error in KB-LLM extraction: {e}")
            return []
    
    def validate_triple(self, triple: Dict[str, Any]) -> bool:
        """
        Validate if a triple conforms to the schema.
        
        Args:
            triple: Triple to validate
            
        Returns:
            Whether the triple is valid according to schema
        """
        # Check if subject predicate and object exist
        if not all(k in triple for k in ["subject", "predicate", "object"]):
            return False
        
        # Check if the predicate exists in relationship_types
        if triple["predicate"] not in self.schema.get("relationship_types", []):
            return False
        
        # More complex validation would check entity types and ranges
        # This is a simplified implementation
        
        return True
    
    def add_triples(self, triples: List[Dict[str, Any]], validate: bool = True) -> int:
        """
        Add triples to the knowledge graph.
        
        Args:
            triples: List of triples to add
            validate: Whether to validate triples against schema
            
        Returns:
            Number of triples added
        """
        added_count = 0
        
        for triple in triples:
            # Validate triple if required
            if validate and not self.validate_triple(triple):
                logger.warning(f"Skipping invalid triple: {triple}")
                continue
            
            # Extract components
            subject = triple["subject"]
            predicate = triple["predicate"]
            obj = triple["object"]
            
            # Add nodes if they don't exist
            if not self.graph.has_node(subject):
                self.graph.add_node(subject)
                self.stats["entity_count"] += 1
            
            if not self.graph.has_node(obj):
                self.graph.add_node(obj)
                self.stats["entity_count"] += 1
            
            # Add edge (relationship)
            self.graph.add_edge(subject, obj, 
                               relation=predicate,
                               confidence=triple.get("confidence", 0.5),
                               source=triple.get("source", "unknown"))
            
            added_count += 1
            self.stats["relationship_count"] += 1
        
        logger.info(f"Added {added_count} triples to knowledge graph")
        return added_count
    
    def query_graph(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query the security knowledge graph.
        
        Args:
            query: Query parameters (e.g., entity, relation type)
            
        Returns:
            List of matching triples
        """
        results = []
        
        # Extract query parameters
        subject = query.get("subject")
        predicate = query.get("predicate")
        obj = query.get("object")
        
        # Find matching edges
        for s, o, data in self.graph.edges(data=True):
            relation = data.get("relation")
            
            # Check if the edge matches the query
            subject_match = subject is None or s == subject
            predicate_match = predicate is None or relation == predicate
            object_match = obj is None or o == obj
            
            if subject_match and predicate_match and object_match:
                results.append({
                    "subject": s,
                    "predicate": relation,
                    "object": o,
                    "confidence": data.get("confidence", 0.5),
                    "source": data.get("source", "unknown")
                })
        
        return results
    
    def get_secure_alternative(self, api: str) -> Optional[str]:
        """
        Get the secure alternative for a potentially insecure API.
        
        Args:
            api: The API to check
            
        Returns:
            Secure alternative if available, None otherwise
        """
        # First check for direct secure alternative
        results = self.query_graph({
            "subject": api,
            "predicate": "hasSecureAlternative"
        })
        
        if results:
            return results[0]["object"]
        
        # Check for replacement (deprecation case)
        results = self.query_graph({
            "subject": api,
            "predicate": "replacedBy"
        })
        
        if results:
            return results[0]["object"]
        
        return None
    
    def get_security_risk(self, api: str) -> Optional[Dict[str, Any]]:
        """
        Get the security risk information for an API.
        
        Args:
            api: The API to check
            
        Returns:
            Security risk information if available
        """
        # Check if API has known vulnerabilities
        vuln_results = self.query_graph({
            "subject": api,
            "predicate": "hasVulnerability"
        })
        
        if not vuln_results:
            return None
        
        # Get vulnerability ID
        vuln_id = vuln_results[0]["object"]
        
        # Get risk level
        risk_results = self.query_graph({
            "subject": vuln_id,
            "predicate": "hasRiskLevel"
        })
        
        risk_level = risk_results[0]["object"] if risk_results else "Unknown"
        
        return {
            "api": api,
            "vulnerability_id": vuln_id,
            "risk_level": risk_level,
            "secure_alternative": self.get_secure_alternative(api)
        }
    
    def save_graph(self, output_path: str) -> bool:
        """
        Save the knowledge graph to a file.
        
        Args:
            output_path: Path to save the graph
            
        Returns:
            Whether the save was successful
        """
        try:
            # Convert graph to serializable format
            graph_data = {
                "nodes": list(self.graph.nodes()),
                "edges": [
                    {
                        "source": s,
                        "target": t,
                        "relation": data.get("relation"),
                        "confidence": data.get("confidence", 0.5),
                        "source_doc": data.get("source", "unknown")
                    }
                    for s, t, data in self.graph.edges(data=True)
                ],
                "schema": self.schema,
                "stats": self.stats
            }
            
            with open(output_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
                
            logger.info(f"Saved knowledge graph to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {e}")
            return False
    
    def load_graph(self, input_path: str) -> bool:
        """
        Load the knowledge graph from a file.
        
        Args:
            input_path: Path to load the graph from
            
        Returns:
            Whether the load was successful
        """
        try:
            with open(input_path, 'r') as f:
                graph_data = json.load(f)
            
            # Create new graph
            self.graph = nx.DiGraph()
            
            # Add nodes
            for node in graph_data.get("nodes", []):
                self.graph.add_node(node)
            
            # Add edges
            for edge in graph_data.get("edges", []):
                self.graph.add_edge(
                    edge["source"],
                    edge["target"],
                    relation=edge["relation"],
                    confidence=edge.get("confidence", 0.5),
                    source=edge.get("source_doc", "unknown")
                )
            
            # Load schema and stats
            self.schema = graph_data.get("schema", self.schema)
            self.stats = graph_data.get("stats", self.stats)
            
            logger.info(f"Loaded knowledge graph from {input_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "entity_count": self.stats["entity_count"],
            "relationship_count": self.stats["relationship_count"],
            "source_count": self.stats["source_count"],
            "validation_count": self.stats["validation_count"],
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges()
        }

def main():
    # Example usage
    schema_path = "data/security_schema.json"
    builder = SecurityKnowledgeGraphBuilder(schema_path)
    
    # Extract from multiple sources
    api_triples = builder.extract_from_api_docs("data/python_deprecations.txt")
    vuln_triples = builder.extract_from_vulnerability_db("data/vulnerabilities.json")
    guide_triples = builder.extract_from_coding_guidelines("data/secure_coding_guidelines.txt")
    
    # Add all triples to graph
    builder.add_triples(api_triples + vuln_triples + guide_triples)
    
    # Save the graph
    builder.save_graph("output/security_knowledge_graph.json")
    
    # Print statistics
    print(f"Knowledge Graph Statistics: {builder.get_stats()}")

if __name__ == "__main__":
    main()
