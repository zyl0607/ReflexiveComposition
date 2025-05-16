import json
from typing import List, Dict, Optional

def run_query_loop(
    rc,
    queries_path: str,
    template_with_kg: str,
    template_without_kg: str,
    context_label: str = "Facts",
    save_results: Optional[str] = None
):
    """
    Run a list of queries with and without knowledge grounding.

    Args:
        rc: ReflexiveComposition instance
        queries_path: Path to .jsonl file with queries
        template_with_kg: Prompt template when grounded=True
        template_without_kg: Prompt template when grounded=False
        context_label: Label for the context block (e.g., 'Facts', 'Context...')
        save_results: Optional path to save results as JSONL
    """
    results = []

    with open(queries_path) as f:
        for line in f:
            entry = json.loads(line)
            query = entry["query"]
            qid = entry.get("id", query[:30].replace(" ", "_"))

            for grounded, template, mode in [
                (False, template_without_kg, "no_kg"),
                (True, template_with_kg, "with_kg")
            ]:
                print(f"\n[ID: {qid}] [{mode}] {query}")
                result = rc.generate_response(
                    query=query,
                    grounded=grounded,
                    template=template,
                    context_label=context_label
                )

                output = {
                    "id": qid,
                    "mode": mode,
                    "query": query,
                    "text": result["text"],
                    "meta": result.get("meta", {})
                }

                print("→", output["text"])
                results.append(output)

    if save_results:
        with open(save_results, "w") as out:
            for item in results:
                out.write(json.dumps(item) + "\n")

    return results
