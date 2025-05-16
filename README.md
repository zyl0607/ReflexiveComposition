# Reflexive Composition

A framework for bidirectional enhancement of Large Language Models (LLMs) and Knowledge Graphs.

## Overview

Reflexive Composition is a modular framework that enables dynamic knowledge integration between LLMs and structured knowledge graphs. It creates a closed-loop system where LLMs assist in knowledge extraction and structured representation, while knowledge graphs provide factual grounding and context for LLM inference.

The framework consists of three core components:

1. **LLM2KG**: LLM-based knowledge extraction and graph construction
2. **HITL**: Human-in-the-loop validation framework
3. **KG2LLM**: Knowledge graph enhanced LLM inference

## Architecture

![Reflexive Composition Architecture](docs/images/architecture.png)

The framework creates a reflexive loop:
- LLMs extract structured knowledge from text (LLM2KG)
- Human experts validate critical knowledge points (HITL)
- Validated knowledge grounds LLM responses (KG2LLM)
- Feedback from responses can trigger knowledge updates

## Installation

```bash
# From PyPI (coming soon)
pip install reflexive-composition

# From source
git clone https://github.com/SystemTwoAI/ReflexiveComposition.git
cd reflexive-composition
pip install -e .
```

## Usage

### Basic Example

```python
from reflexive_composition.core import ReflexiveComposition

# Initialize the framework
rc = ReflexiveComposition(
    kb_llm_config={
        "model_name": "gpt-3.5-turbo",
        "api_key": "your-openai-api-key",
        "model_provider": "openai"
    },
    target_llm_config={
        "model_name": "gpt-3.5-turbo",
        "api_key": "your-openai-api-key",
        "model_provider": "openai"
    },
    kg_config={
        "storage_type": "in_memory",
        "schema": {
            "entity_types": ["Person", "Event", "Location"],
            "relationship_types": ["LocatedIn", "ParticipatedIn"],
            "version": 1
        }
    }
)

# Extract knowledge from text
extraction_result = rc.extract_knowledge(
    source_text="Donald Trump survived an assassination attempt during a rally in Butler, Pennsylvania on July 13, 2024.",
    schema=rc.knowledge_graph.schema
)

# Update knowledge graph
rc.update_knowledge_graph(extraction_result["triples"])

# Generate enhanced response
response = rc.generate_response(
    query="What happened to Donald Trump at the rally?",
    retrieve_context=True
)

print(response["text"])
```

See the [examples](examples/) directory for more detailed usage examples.

## Case Studies

The framework has been evaluated across several case studies:

1. **Temporal Knowledge Management**: Handling evolving facts and current events
2. **Private Data Integration**: Secure integration of private/sensitive information
3. **Historical Bias Mitigation**: Reducing inherited biases in code generation

## Features

- **Dynamic Knowledge Extraction**: Extract structured knowledge from text using LLMs
- **Schema Evolution**: Automatically suggest and manage schema updates
- **Strategic Validation**: Focus human oversight on critical decision points
- **Knowledge-Enhanced Generation**: Ground LLM responses in validated knowledge
- **Contradiction Detection**: Identify and resolve conflicts between LLM outputs and knowledge
- **Multiple Storage Backends**: Support for in-memory, RDF, and Neo4j graph storage

## Development

### Prerequisites

- Python 3.8+
- Required packages (see [pyproject.toml](pyproject.toml))

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/SystemTwoAI/ReflexiveComposition.git
cd reflexive-composition

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## License

This project is licensed under the GNU v3 license.

## Citation

If you use Reflexive Composition in your research, please cite:

```bibtex
@misc{author2025reflexive,
  title={Reflexive Composition: Bidirectional Enhancement of Language Models and Knowledge Graphs},
  author={Virendra Mehta},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/SystemTwoAI/ReflexiveComposition}}
}
```

## Acknowledgments

This framework is based on research in neural-symbolic integration, knowledge graph engineering, and large language models.
