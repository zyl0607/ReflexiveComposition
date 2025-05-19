"""
Evaluator for Reflexive Composition outputs.

This base class provides a reusable interface for evaluating LLM generations
across multiple case studies (e.g., code security, medical, temporal grounding).
"""

from typing import List, Dict
import pandas as pd
import json

class Evaluator:
    """
    Base class for evaluating generated outputs. Meant to be subclassed for case-specific logic.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}

    def evaluate_outputs(self, outputs: List[Dict]) -> pd.DataFrame:
        """
        Generic evaluation logic. Override in subclass to apply domain-specific scoring.

        Args:
            outputs: List of generation output records.

        Returns:
            pd.DataFrame with basic evaluation stats.
        """
        records = []

        for entry in outputs:
            record = {
                "id": entry.get("id"),
                "output_length": len(entry.get("output", "")),
                "has_output": bool(entry.get("output")),
                # Extend this with more generic or domain-specific checks
            }
            records.append(record)

        return pd.DataFrame(records)

    @staticmethod
    def load_json_file(path: str) -> Dict:
        """
        Load and parse a JSON file. Returns empty dict if there's an error.
        """
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON from {path}: {e}")
            return {}

    @staticmethod
    def normalize_output(output: str) -> str:
        """
        Normalize generated output (e.g., strip whitespace, dedent).
        """
        return output.strip()