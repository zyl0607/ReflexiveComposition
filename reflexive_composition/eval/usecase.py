from abc import ABC, abstractmethod
from typing import List, Dict
import os

class UseCase(ABC):
    """
    Abstract base class for Reflexive Composition use cases.
    Defines the interface and shared functionality.
    """

    def __init__(self):
        self.tasks = []
        self.outputs = []

    def default_llm_config(self):
        return {
            "model_name": os.environ.get("KB_LLM_MODEL", "gemini-2.0-flash"),
            "api_key": os.environ.get("GEMINI_API_KEY"),
            "model_provider": "google",
        }

    @abstractmethod
    def load_tasks(self, path: str) -> List[Dict]:
        pass

    @abstractmethod
    def extract_and_generate(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    def run_full_pipeline(self):
        self.extract_and_generate()
        return self.evaluate()
