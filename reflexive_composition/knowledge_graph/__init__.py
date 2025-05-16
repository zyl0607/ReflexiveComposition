# reflexive_composition/knowledge_graph/__init__.py
"""
Knowledge Graph module for storing, querying, and updating structured knowledge.
"""

from .graph import KnowledgeGraph
from .subgraph import SubgraphRetriever
from .storage import GraphStorage

__all__ = ['KnowledgeGraph', 'SubgraphRetriever', 'GraphStorage']