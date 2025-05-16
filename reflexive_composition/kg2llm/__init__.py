# reflexive_composition/kg2llm/__init__.py
"""
KG2LLM module for knowledge graph enhanced LLM inference.
"""

from .target_llm import TargetLLM
from .prompt_builder import PromptBuilder
from .contradiction_detector import ContradictionDetector

__all__ = ['TargetLLM', 'PromptBuilder', 'ContradictionDetector']