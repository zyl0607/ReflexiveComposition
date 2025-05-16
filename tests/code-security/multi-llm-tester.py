from typing import Dict, List
from langchain.llms import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
import torch

class MultiLLMTester:
    def __init__(self):
        # Initialize different LLMs
        self.llms = self._initialize_llms()
        
        # Test cases focusing on real-world usage
        self.python_cases = [
            {
                "prompt": "Write a function to get current UTC time",
                "deprecated": "datetime.utcnow",
                "replacement": "datetime.now(tz=datetime.UTC)",
                "category": "Time Operations"
            },
            {
                "prompt": "Create code to parse a numeric AST node",
                "deprecated": "ast.Num",
                "replacement": "ast.Constant",
                "category": "AST Operations"
            },
            {
                "prompt": "Write code to check thread count",
                "deprecated": "threading.activeCount",
                "replacement": "threading.active_count",
                "category": "Threading"
            }
        ]
        
        self.cobol_cases = [
            {
                "prompt": "Write a COBOL program to modify paragraph execution sequence",
                "deprecated": "ALTER",
                "replacement": "PERFORM",
                "category": "Control Flow"
            },
            {
                "prompt": "Create a COBOL program for currency calculations",
                "deprecated": ["PICTURE 9V99", "DECIMAL-POINT IS COMMA"],
                "replacement": ["PICTURE 9(7)V99", "DECIMAL-POINT IS PERIOD"],
                "category": "Financial"
            },
            {
                "prompt": "Write a COBOL program to handle date operations",
                "deprecated": ["ACCEPT DATE", "ACCEPT TIME"],
                "replacement": ["FUNCTION CURRENT-DATE"],
                "category": "Date Handling"
            }
        ]

    def _initialize_llms(self) -> Dict:
        """Initialize different LLMs for testing"""
        llms = {}
        
        # CodeLlama
        try:
            tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python")
            model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Python")
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
            llms["CodeLlama"] = HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            print(f"Failed to load CodeLlama: {e}")
        
        # StarCoder2
        try:
            tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")
            model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder2-3b")
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
            llms["StarCoder2"] = HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            print(f"Failed to load StarCoder2: {e}")
        
        # Claude (via API)
        try:
            llms["Claude"] = ChatOpenAI(model="claude-3-haiku-20240307")
        except Exception as e:
            print(f"Failed to load Claude: {e}")
        
        return llms

    def check_deprecated_usage(self, code: str, deprecated: str | List[str]) -> bool:
        """Check if code contains deprecated features"""
        if isinstance(deprecated, str):
            deprecated = [deprecated]
        
        return any(dep.lower() in code.lower() for dep in deprecated)

    def run_tests(self, language: str = "python") -> pd.DataFrame:
        """Run tests across all LLMs"""
        results = []
        cases = self.python_cases if language.lower() == "python" else self.cobol_cases
        
        for llm_name, llm in self.llms.items():
            print(f"\nTesting {llm_name} on {language}...")
            
            for case in cases:
                prompt = f"""Generate {language} code for the following task. 
                Return only the code without explanations:
                {case['prompt']}"""
                
                try:
                    # Generate code
                    response = llm.predict(prompt)
                    
                    # Check for deprecated usage
                    uses_deprecated = self.check_deprecated_usage(
                        response, case["deprecated"]
                    )
                    
                    results.append({
                        "LLM": llm_name,
                        "Language": language,
                        "Category": case["category"],
                        "Prompt": case["prompt"],
                        "Uses_Deprecated": uses_deprecated,
                        "Deprecated_Feature": str(case["deprecated"]),
                        "Replacement": str(case["replacement"]),
                        "Generated_Code": response
                    })
                    
                except Exception as e:
                    print(f"Error with {llm_name} on {case['prompt']}: {e}")
        
        return pd.DataFrame(results)

def analyze_results(df: pd.DataFrame) -> None:
    """Analyze and print test results"""
    print("\nDeprecation Usage Analysis:")
    print("===========================")
    
    # Overall stats by LLM and language
    print("\nDeprecation rates by LLM and Language:")
    pivot = pd.pivot_table(
        df,
        values='Uses_Deprecated',
        index=['LLM'],
        columns=['Language'],
        aggfunc='mean'
    )
    print(pivot * 100)
    
    # Category analysis
    print("\nDeprecation rates by Category:")
    category_stats = df.groupby(['Language', 'Category'])['Uses_Deprecated'].mean()
    print(category_stats * 100)
    
    # Save detailed results
    df.to_csv('llm_deprecation_analysis.csv', index=False)

def main():
    tester = MultiLLMTester()
    
    # Run tests for both languages
    python_results = tester.run_tests("python")
    cobol_results = tester.run_tests("cobol")
    
    # Combine and analyze results
    all_results = pd.concat([python_results, cobol_results])
    analyze_results(all_results)

if __name__ == "__main__":
    main()