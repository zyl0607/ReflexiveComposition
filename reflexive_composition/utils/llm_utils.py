import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def shorten_response(text: str, max_len: int = 300) -> str:
    return text.strip()[:max_len].rsplit(" ", 1)[0] + "..." if len(text) > max_len else text

def format_triples(triples: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"- {t['subject']} {t['predicate']} {t['object']}"
        for t in triples if all(k in t for k in ("subject", "predicate", "object"))
    )

def count_tokens(text: str) -> int:
    # Naive token count approximation by whitespace
    return len(text.split())

def log_result(result: Dict[str, Any], prefix: str = ""):
    query = result.get("query") or result.get("meta", {}).get("query", "[unknown query]")
    print(f"\n{prefix}Query: {query}")
    print("→", shorten_response(result["text"]))
    if result.get("meta", {}).get("has_contradictions"):
        print("⚠ Contradictions detected")
    print("Tokens used:", result.get("meta", {}).get("token_count", "N/A"))
    print("-" * 60)

def attach_prompt_stats(prompt: str, response_meta: Dict[str, Any]):
    """
    Add token-level metadata from prompt text.
    """
    response_meta["prompt_tokens"] = count_tokens(prompt)

def extract_text(response: dict, provider: str) -> str:
    
    if isinstance(response, dict) and "text" in response:
        return response["text"]

    provider = provider.strip().lower()

    try:
        if provider == "openai":
            return response["choices"][0]["message"]["content"]
        elif provider == "anthropic":
            return response["completion"]
        elif provider == "cohere":
            return response["generations"][0]["text"]
        elif provider == "google":
            return response["candidates"][0]["content"]["parts"][0]["text"]
        elif isinstance(response, dict) and "text" in response:
            return response["text"]
        else:
            return str(response)
    except Exception:
        return str(response)
