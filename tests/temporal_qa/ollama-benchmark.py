#from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import PromptTemplate
#from langchain_core.output_parsers import StrOutputParser
#from langchain.chains import LLMChain
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import json
import os
from datetime import datetime

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark models and prompts"""
    models: Dict[str, Dict]  # Model configurations
    system_prompt: str
    query_template: str
    output_format: str

class PresidentsBenchmark:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.models = self._initialize_models()
        self.prompt = PromptTemplate(
            input_variables=["question"],
            template=f"{self.config.system_prompt}\n\n{{question}}"
        )

    def _initialize_models(self) -> Dict:
        """Initialize local models using Ollama"""
        models = {}
        for model_name, config in self.config.models.items():
            try:
                models[model_name] = OllamaLLM(
                    model=config["model_name"],
                    temperature=0.0,
                    num_gpu=1 if config.get("use_gpu", True) else 0
                )
                print(f"Successfully loaded {model_name}")
            except Exception as e:
                print(f"Failed to load {model_name}: {str(e)}")
        return models

    def evaluate_model(self, model_name: str, facts: List[Dict]) -> Dict:
        """Evaluate a single model on benchmark facts"""
        model = self.models[model_name]
        chain = self.prompt | model
        
        results = {
            "correct": 0,
            "outdated": 0,
            "irrelevant": 0,
            "total": 0,
            "responses": []  # Store actual responses for analysis
        }

        for fact in facts:
            for query in fact["queries"][:3]:  # Use first 3 queries per fact
                try:
                    response = chain.invoke({ "question" : query})
                    result = self._validate_response(response, fact)
                    results[result] += 1
                    results["total"] += 1
                    results["responses"].append({
                        "query": query,
                        "response": response,
                        "evaluation": result,
                        "fact": fact["id"]
                    })
                    print(f"Query: {query}\nResponse: {response}\nEvaluation: {result}\n")
                except Exception as e:
                    print(f"Error with {model_name} on query '{query}': {str(e)}")

        return results

    def _validate_response(self, response: str, fact: Dict) -> str:
        """Validate model response against fact with category-aware matching logic"""
        if not response or not isinstance(response, str):
            return "irrelevant"
            
        response = response.lower().strip()
        current = fact["current_answer"].lower()
        category = fact.get("category", "").lower()
        
        # Get key phrases from fact definition
        key_phrases = fact.get("key_phrases", [])
        if not key_phrases:  # Fallback to splitting current answer
            key_phrases = [phrase.strip() for phrase in current.split(",")]
        
        # Count phrase matches
        content_matches = sum(1 for phrase in key_phrases if phrase in response)
        
        # Adjust required matches based on category
        if category == "basic":
            required_matches = 1  # Basic facts need fewer matches
        elif category == "security":
            required_matches = max(2, len(key_phrases) // 2)  # Security needs more strict matching
        else:
            required_matches = max(1, len(key_phrases) // 3)  # Default threshold
        
        # Check if temporal validation is required
        requires_temporal = fact.get("requires_temporal", category not in ["basic"])
        
        if requires_temporal:
            fact_date = fact.get("date", "")
            fact_year = fact_date[:4] if fact_date else ""
            
            temporal_match = False
            if fact_year and fact_year in response:
                temporal_match = True
            elif fact_date:
                try:
                    date_obj = datetime.strptime(fact_date, "%Y-%m-%d")
                    month = date_obj.strftime("%B").lower()
                    day = str(date_obj.day)
                    if month in response and day in response:
                        temporal_match = True
                except ValueError:
                    pass
            
            if temporal_match and content_matches >= required_matches:
                return "correct"
        else:
            # For facts not requiring temporal validation
            if content_matches >= required_matches:
                return "correct"
        
        # Check for outdated information
        for old_answer in fact.get("temporal_answers", []):
            old_text = old_answer["text"].lower()
            if any(phrase in response for phrase in old_text.split()):
                return "outdated"
        
        return "irrelevant"

# Example configuration for local models
default_config = BenchmarkConfig(
    models={
         "Llama-3.2": {
            "model_name": "llama3.2:latest",
            "year": "2024",
            "use_gpu": True
        },
         "Llama-3": {
            "model_name": "llama3:instruct",
            "year": "2023",
            "use_gpu": True
        #},
       # "CodeLlama-Instruct": {
        #    "model_name": "codellama:13b-instruct",
         #   "year": "2023",
          #  "use_gpu": True
        #},
        #"Llama-2": {
        #    "model_name": "llama2:13b-chat",
        #    "year": "2023",
        #    "use_gpu": True
        #},
        #"Mistral": {
        #    "model_name": "mistral",
        #    "year": "2023",
        #    "use_gpu": True
        #},
        #"Mixtral": {
            #"model_name": "mixtral",
            #"year": "2023",
            #"use_gpu": True
        #},
        #"StarCoder2": {
            #"model_name": "starcoder2",
            #"year": "2024",
            #"use_gpu": True
        }
    },
    system_prompt="""You are answering questions about US Presidents. 
    Provide accurate, up-to-date information. If you're not sure about current information,
    acknowledge the uncertainty. Base your answers only on verifiable facts.""",
    query_template="{question}",
    output_format="natural"
)

def main():
    # Load benchmark facts
    with open("recency_benchmark_facts.json", "r") as f:
        facts = json.load(f)["facts"]

    # Initialize benchmark
    benchmark = PresidentsBenchmark(default_config)
    
    # Run evaluations
    results = []
    responses_by_model = {}
    
    for model_name, model_config in benchmark.config.models.items():
        print(f"\nEvaluating {model_name}...")
        model_results = benchmark.evaluate_model(model_name, facts)
        
        # Store aggregate results
        if (model_results["total"] != 0):
            results.append({
                "model": model_name,
                "year": model_config["year"],
                "correct": f"{(model_results['correct']/model_results['total'])*100:.1f}%",
                "outdated": f"{(model_results['outdated']/model_results['total'])*100:.1f}%",
                "irrelevant": f"{(model_results['irrelevant']/model_results['total'])*100:.1f}%"
            })
        else:    
            print("Error: no results")
        
        # Store detailed responses
        responses_by_model[model_name] = model_results["responses"]

    # Save aggregate results
    results_df = pd.DataFrame(results)
    results_df.to_csv("benchmark_results.csv", index=False)
    
    # Save detailed responses
    with open("detailed_responses.json", "w") as f:
        json.dump(responses_by_model, f, indent=2)
    
    print("\nAggregate Results:")
    print(results_df.to_string())
    print("\nDetailed responses saved to detailed_responses.json")

if __name__ == "__main__":
    main()
