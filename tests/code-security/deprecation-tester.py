from typing import List, Dict
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import re

class DeprecationTester:
    def __init__(self):
        # Initialize LLM - we'll use a modern code-focused model
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        # Test cases that commonly trigger deprecated APIs
        self.python_test_cases = [
            {
                "prompt": "Create a Python function to get the current UTC time",
                "deprecated": "datetime.utcnow",
                "replacement": "datetime.now(tz=datetime.UTC)",
                "risk": "Timezone handling inconsistencies"
            },
            {
                "prompt": "Write a function to check if a thread is active",
                "deprecated": "threading.activeCount",
                "replacement": "threading.active_count",
                "risk": "Thread synchronization issues"
            },
            {
                "prompt": "Write code to analyze a numerical constant in Python AST",
                "deprecated": "ast.Num",
                "replacement": "ast.Constant",
                "risk": "Parser compatibility issues"
            }
        ]
        
        self.cobol_test_cases = [
            {
                "prompt": "Write a COBOL program to modify control flow based on a condition",
                "deprecated": "ALTER",
                "replacement": "PERFORM",
                "risk": "Unpredictable control flow"
            },
            {
                "prompt": "Create a COBOL program to handle monetary calculations",
                "deprecated": "PICTURE 9V99",
                "replacement": "PICTURE 9(7)V99",
                "risk": "Financial calculation precision"
            }
        ]

    def check_for_deprecated_usage(self, code: str, deprecated_api: str) -> bool:
        """Check if code contains deprecated API usage"""
        return bool(re.search(rf'\b{re.escape(deprecated_api)}\b', code))

    def test_llm_responses(self, language: str = "python") -> Dict[str, float]:
        """Test LLM responses for deprecated API usage"""
        test_cases = (self.python_test_cases if language.lower() == "python" 
                     else self.cobol_test_cases)
        
        results = {
            "total_tests": len(test_cases),
            "deprecated_usage": 0,
            "test_details": []
        }
        
        for test in test_cases:
            # Create prompt with safety emphasis
            prompt = ChatPromptTemplate.from_template(
                """You are a coding assistant. Write code for the following task, 
                focusing on modern and secure implementations:
                
                {task}
                
                Return only the code, no explanations."""
            )
            
            # Get LLM response
            messages = prompt.format_messages(task=test["prompt"])
            response = self.llm.predict(messages[0].content)
            
            # Check for deprecated API usage
            uses_deprecated = self.check_for_deprecated_usage(
                response, test["deprecated"]
            )
            
            if uses_deprecated:
                results["deprecated_usage"] += 1
            
            results["test_details"].append({
                "prompt": test["prompt"],
                "deprecated_api": test["deprecated"],
                "replacement": test["replacement"],
                "used_deprecated": uses_deprecated,
                "security_risk": test["risk"],
                "generated_code": response
            })
        
        results["deprecation_rate"] = (results["deprecated_usage"] / 
                                     results["total_tests"])
        
        return results

def main():
    tester = DeprecationTester()
    
    # Test Python code generation
    print("\nTesting Python code generation...")
    python_results = tester.test_llm_responses("python")
    print(f"Deprecation rate: {python_results['deprecation_rate']*100:.1f}%")
    
    # Test COBOL code generation
    print("\nTesting COBOL code generation...")
    cobol_results = tester.test_llm_responses("cobol")
    print(f"Deprecation rate: {cobol_results['deprecation_rate']*100:.1f}%")
    
    # Print detailed results for deprecated usages
    print("\nDetailed analysis of deprecated API usage:")
    for language, results in [("Python", python_results), ("COBOL", cobol_results)]:
        print(f"\n{language} Results:")
        for test in results["test_details"]:
            if test["used_deprecated"]:
                print(f"""
                Task: {test['prompt']}
                Deprecated API: {test['deprecated']}
                Security Risk: {test['security_risk']}
                Generated Code:
                {test['generated_code']}
                """)

if __name__ == "__main__":
    main()