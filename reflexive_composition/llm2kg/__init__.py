# reflexive_composition/llm2kg/__init__.py
"""
LLM2KG module for extracting structured knowledge from text using LLMs.
"""

from .knowledge_builder import KnowledgeBuilderLLM
from .extractor import KnowledgeExtractor
from .schema_evolution import SchemaManager

__all__ = ['KnowledgeBuilderLLM', 'KnowledgeExtractor', 'SchemaManager']
