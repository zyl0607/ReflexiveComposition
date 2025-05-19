import json
from typing import List, Dict, Tuple
from sklearn.metrics import precision_recall_fscore_support


def evaluate_extraction(gold_triples: List[Tuple], predicted_triples: List[Tuple]) -> Dict[str, float]:
    """
    Evaluate LLM2KG extraction using F1, precision, recall.
    Triples are assumed to be (subject, predicate, object) tuples.
    """
    gold_set = set(gold_triples)
    predicted_set = set(predicted_triples)

    tp = len(gold_set & predicted_set)
    fp = len(predicted_set - gold_set)
    fn = len(gold_set - predicted_set)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_hitl_intervention(all_cases: List[Dict]) -> Dict[str, float]:
    """
    Evaluate HITL effectiveness by measuring override rate and validator intervention frequency.
    """
    total_cases = len(all_cases)
    interventions = sum(1 for case in all_cases if case.get("validator_override"))
    overrides = sum(1 for case in all_cases if case.get("validator_override") and case.get("llm_incorrect"))

    return {
        "intervention_rate": interventions / total_cases if total_cases else 0.0,
        "override_precision": overrides / interventions if interventions else 0.0
    }


def evaluate_generation_accuracy(gold_answers: List[str], generated_answers: List[str]) -> Dict[str, float]:
    """
    Naive exact-match metric for generation accuracy (e.g., temporal QA or medical response).
    """
    total = len(gold_answers)
    correct = sum(1 for gold, pred in zip(gold_answers, generated_answers) if gold.strip() == pred.strip())

    return {"accuracy": correct / total if total else 0.0}


def load_jsonl(path: str) -> List[Dict]:
    with open(path, 'r') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]