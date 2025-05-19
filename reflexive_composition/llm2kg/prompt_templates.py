# reflexive_composition/llm2kg/prompt_templates.py
"""
Prompt templates for knowledge extraction and schema generation.

This module provides standardized prompts for guiding LLMs in extracting
structured knowledge and generating or updating knowledge graph schemas.
"""

from typing import Dict, List, Any, Optional


def get_extraction_prompt(text: str, schema: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate a general knowledge extraction prompt.
    
    Args:
        text: Source text for extraction
        schema: Optional schema to guide extraction
        
    Returns:
        Formatted prompt
    """
    schema_guidance = ""
    if schema:
        entity_types = ", ".join(schema.get("entity_types", []))
        relationship_types = ", ".join(schema.get("relationship_types", []))
        schema_guidance = f"""
Schema information:
Entity types: {entity_types}
Relationship types: {relationship_types}

Make sure all extracted information follows this schema.
"""
        
    prompt = f"""Extract structured knowledge from the following text.
Return the extracted information as JSON with entities, relationships, and attributes.

Text: {text}

{schema_guidance}

Expected output format:
{{
    "triples": [
        {{
            "subject": "entity_name",
            "predicate": "relation_type",
            "object": "target_entity_or_value",
            "confidence": 0.95
        }}
    ]
}}

Remember:
1. All subjects and objects should be specific entities, not general concepts
2. All predicates should be specific relationship types, and use "attribute" for properties
3. Include a confidence score between 0.0 and 1.0 for each triple
4. Only extract information explicitly stated in the text
5. Be sure to include dates, locations, and important attributes
"""
    
    return prompt


def get_temporal_extraction_prompt(text: str, schema: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate a prompt specifically for temporal knowledge extraction.
    
    Args:
        text: Source text for extraction
        schema: Optional schema to guide extraction
        
    Returns:
        Formatted prompt
    """
    schema_guidance = ""
    if schema:
        entity_types = ", ".join(schema.get("entity_types", []))
        relationship_types = ", ".join(schema.get("relationship_types", []))
        schema_guidance = f"""
Schema information:
Entity types: {entity_types}
Relationship types: {relationship_types}

Make sure all extracted information follows this schema.
"""
    
    prompt = f"""Extract temporal knowledge and events from the following text.
Focus on entities, relationships, and especially time-related information.

Text: {text}

{schema_guidance}

Expected output format:
{{
    "triples": [
        {{
            "subject": "entity_name",
            "predicate": "relation_type",
            "object": "target_entity_or_value",
            "confidence": 0.95,
            "temporal_context": {{
                "start_time": "YYYY-MM-DD",
                "end_time": "YYYY-MM-DD",
                "is_current": true/false
            }}
        }}
    ]
}}

Pay special attention to:
1. Date and time information (convert to YYYY-MM-DD format where possible)
2. Event sequences and their temporal relationships
3. Current vs. historical states of entities
4. Only extract information explicitly stated in the text
5. Include "attribute" as predicate for properties like dates and descriptions
"""
    
    return prompt


def get_domain_specific_extraction_prompt(
    text: str, 
    domain: str,
    schema: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a domain-specific extraction prompt.
    
    Args:
        text: Source text for extraction
        domain: Domain name (e.g., "medical", "financial", "news")
        schema: Optional schema to guide extraction
        
    Returns:
        Formatted prompt
    """
    # Domain-specific guidance
    domain_guidance = {
        "medical": """
Focus on extracting:
- Patient details and demographics
- Medical conditions and diagnoses
- Treatments and medications
- Healthcare providers and facilities
- Temporal relationships between medical events
""",
        "financial": """
Focus on extracting:
- Financial entities (companies, markets, currencies)
- Economic indicators and metrics
- Financial events and transactions
- Stakeholders and their relationships
- Temporal aspects of financial information
""",
        "news": """
Focus on extracting:
- People, organizations, and locations mentioned in news
- Events and their details (when, where, who was involved)
- Causal relationships between events
- Statements, quotes, and their sources
- Temporal ordering of news events
"""
    }.get(domain.lower(), "")
    
    schema_guidance = ""
    if schema:
        entity_types = ", ".join(schema.get("entity_types", []))
        relationship_types = ", ".join(schema.get("relationship_types", []))
        schema_guidance = f"""
Schema information:
Entity types: {entity_types}
Relationship types: {relationship_types}

Make sure all extracted information follows this schema.
"""
    
    prompt = f"""Extract structured knowledge from the following {domain} text.
Return the extracted information as JSON with entities, relationships, and attributes.

Text: {text}

{domain_guidance}

{schema_guidance}

Expected output format:
{{
    "triples": [
        {{
            "subject": "entity_name",
            "predicate": "relation_type",
            "object": "target_entity_or_value",
            "confidence": 0.95
        }}
    ]
}}

Remember:
1. All subjects and objects should be specific entities, not general concepts
2. All predicates should be specific relationship types, and use "attribute" for properties
3. Include a confidence score between 0.0 and 1.0 for each triple
4. Only extract information explicitly stated in the text
5. Be sure to include dates, locations, and important attributes
"""
    
    return prompt


def get_schema_generation_prompt(domain_description: str, examples: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Generate a prompt for schema generation.
    
    Args:
        domain_description: Description of the domain for schema generation
        examples: Optional examples to guide schema generation
        
    Returns:
        Formatted prompt
    """
    examples_text = ""
    if examples:
        examples_text = "Examples from the domain:\n"
        for i, example in enumerate(examples[:5], 1):
            if isinstance(example, dict) and all(k in example for k in ("subject", "predicate", "object")):
                examples_text += f"{i}. {example['subject']} {example['predicate']} {example['object']}\n"
            else:
                examples_text += f"{i}. {str(example)}\n"
    
    prompt = f"""Design a knowledge graph schema for the following domain:

Domain Description: {domain_description}

{examples_text}

Create a schema that includes:
1. Entity types - Classes of objects in the domain
2. Relationship types - Types of relationships between entities
3. Key attributes - Important properties for each entity type

Format the output as JSON with the following structure:
{{
  "entity_types": ["Type1", "Type2", ...],
  "relationship_types": ["Relation1", "Relation2", ...],
  "attributes": [
    {{"entity_type": "Type1", "attributes": ["attr1", "attr2", ...]}},
    ...
  ]
}}

Design principles to follow:
- Keep entity and relationship names clear and concise
- Use consistent naming conventions (e.g., CamelCase for entity types)
- Focus on the core concepts and relationships in the domain
- Include only entity types and relationships that are clearly relevant to the domain
- Consider both common and edge cases in the domain
"""
    
    return prompt


def get_schema_evolution_prompt(current_schema: Dict[str, Any], new_information: str) -> str:
    """
    Generate a prompt for schema evolution.
    
    Args:
        current_schema: Current schema definition
        new_information: New information that might require schema updates
        
    Returns:
        Formatted prompt
    """
    # Format the current schema
    entity_types = ", ".join(current_schema.get("entity_types", []))
    relationship_types = ", ".join(current_schema.get("relationship_types", []))
    
    prompt = f"""Evaluate whether the current knowledge graph schema needs updates based on new information.

Current Schema:
- Entity Types: {entity_types}
- Relationship Types: {relationship_types}

New Information:
{new_information}

Suggest necessary updates to the schema based on the new information.
Format your response as JSON with the following structure:
{{
  "entity_types_to_add": ["NewType1", "NewType2"],
  "relationship_types_to_add": ["NewRelation1", "NewRelation2"],
  "attributes_to_add": [
    {{"entity_type": "ExistingType", "attributes": ["new_attr1", "new_attr2"]}},
    {{"entity_type": "NewType1", "attributes": ["attr1", "attr2"]}}
  ],
  "reasoning": "Explanation of why these updates are necessary"
}}

Important guidelines:
1. Only suggest additions that are clearly needed based on the new information
2. Maintain consistency with the existing schema
3. Use naming conventions consistent with the existing schema
4. Include clear reasoning for each suggested addition
"""
    
    return prompt
