# reflexive_composition/kg2llm/prompt_builder.py
"""
Prompt construction for knowledge graph enhanced LLM inference.

This module handles the construction of prompts that incorporate knowledge
graph context for improved response generation.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Builds prompts for knowledge-enhanced LLM inference.
    """

    def __init__(self,
                 default_template: Optional[str] = None,
                 max_context_length: int = 2000):
        """
        Args:
            default_template: Default prompt template
            max_context_length: Maximum length for context in tokens
        """
        self.max_context_length = max_context_length
        self.default_template = default_template or """You are a helpful assistant.

{context_block}

User query:
{query}

Answer the question. If facts are provided above, prefer them; otherwise, use your own knowledge. If facts are provided but do not contain enough information to answer the query, clearly say so. Do not make up facts.
"""

    def build_prompt(self,
                     query: str,
                     context_label: str = "Facts",
                     context: Optional[List[Union[Tuple[str, str, str], Dict[str, Any]]]] = None,                     template: Optional[str] = None
                     ) -> Tuple[str, Dict[str, Any]]:
        """
        Build a prompt using structured context and an optional custom template.

        Args:
            query: The user question
            context: List of (subject, predicate, object) triples
            template: Optional prompt template to override default

        Returns:
            A tuple of (prompt string, metadata dictionary)
        """
        print(">>> DEBUG: build_prompt received context:", context)
        print(">>> DEBUG: type(context):", type(context), " length:", len(context) if context else 0)
        meta = {
            "context_size": len(context) if context else 0,
            "triples_used": sum(1 for t in context if all(k in t for k in ("subject", "predicate", "object"))) if context else 0,
            "format": "default",
            "context_format": "triples"
        }

        try:
            if context:
                valid_triples = [t for t in context if all(k in t for k in ("subject", "predicate", "object"))]
                meta["triples_used"] = len(valid_triples)
                formatted_context = "\n".join(f"- {t['subject']} {t['predicate']} {t['object']}" for t in valid_triples)
                if meta["triples_used"] == 0:
                    logger.warning("No usable triples found — prompt will be unguided.")
            else:
                formatted_context = ""
                logger.warning("No context provided — prompt will be unguided.")

            context_block = f"{context_label}:\n" + formatted_context.strip() if formatted_context else ""
            prompt_template = template or self.default_template
            
            prompt = prompt_template.format(
                context_block=context_block,
                query=query.strip()
            )

            logger.debug("Prompt built successfully.")
            return prompt.strip(), meta

        except Exception as e:
            logger.error(f"Error building prompt: {e}", exc_info=True)
            return f"Error: failed to build prompt for query: {query}", {
                "error": str(e),
                "context_size": 0,
                "triples_used": 0
            }
