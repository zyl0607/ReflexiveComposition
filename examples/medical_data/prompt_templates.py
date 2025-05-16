WITH_CONTEXT_TEMPLATE = """You are a clinical assistant with access to structured patient information.

{context_block}

Clinical question:
{query}

Use the above information to answer precisely and safely. Do not assume facts that are not explicitly stated."""

NO_CONTEXT_TEMPLATE = """You are a clinical assistant.

Clinical question:
{query}

Answer using your own general knowledge. Do not reference external documents or links."""
