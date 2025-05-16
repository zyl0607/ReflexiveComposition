WITH_CONTEXT_PROMPT_TEMPLATE = """You are a timeline-aware assistant trained to use factual historical context.

{context_block}

Question:
{query}

Answer using only the above facts. If the answer cannot be determined from them, say so."""

NO_CONTEXT_PROMPT_TEMPLATE = """You are a timeline-aware assistant trained to use factual historical context.

Question:
{query}

Answer using your own general knowledge. Do not reference external documents or links."""
