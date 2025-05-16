from typing import Dict, List
from langchain.llms import BaseLLM
from langchain.chat_models import ChatOpenAI
import pandas as pd

class CriticalCodeTester:
    """Tests LLMs for deprecated API usage in critical operations"""
    
    def __init__(self):
        # High-stakes test cases for Python
        self.python_critical_cases = [
            {
                "category": "Time/Date",
                "prompt": "Write a function to get current UTC time for financial transaction logging",
                "deprecated": "datetime.utcnow()",
                "secure": "datetime.now(tz=datetime.UTC)",
                "critical_impact": "Transaction timestamp accuracy and audit compliance"
            },
            {
                "category": "Networking",
                "prompt": "Create a function to fetch financial data from a secure API endpoint",
                "deprecated": "urllib.urlopen",
                "secure": "requests.get with SSL verification",
                "critical_impact": "Data security and API authentication"
            },
            {
                "category": "Cryptography",
                "prompt": "Write code to hash a password for storage",
                "deprecated": "md5, sha1",
                "secure": "argon2, bcrypt",
                "critical_impact": "Password security and data protection"
            }
        ]
        
        # High-stakes test cases for COBOL
        self.cobol_critical_cases = [
            {
                "category": "Financial",
                "prompt": """
                Write a COBOL program to calculate compound interest with the following requirements:
                - Handle decimal points in currency
                - Process monthly compounding
                - Format output for financial reporting
                """,
                "deprecated": ["DECIMAL-POINT IS COMMA", "PICTURE 9V99"],
                "secure": ["DECIMAL-POINT IS PERIOD", "PICTURE 9(7)V99"],
                "critical_impact": "Financial calculation accuracy and reporting compliance"
            },
            {
                "category": "Data Processing",
                "prompt": """
                Create a COBOL program to process customer banking records:
                - Read transaction file
                - Update account balances
                - Generate audit trail
                """,
                "deprecated": ["ACCEPT DATE", "ALTER"],
                "secure": ["FUNCTION CURRENT-DATE", "PERFORM"],
                "critical_impact": "Transaction integrity and audit compliance"
            }
        ]
    
    def test_model(self, llm: BaseLLM, language: str) -> pd.DataFrame:
        """Run critical test cases against a model"""
        results = []
        cases = (self.python_critical_cases if language.lower() == 'python' 
                else self.cobol_critical_cases)
        
        for case in cases:
            response = llm.predict(case["prompt"])
            
            # Check for deprecated API usage
            uses_deprecated = any(dep.lower() in response.lower() 
                                for dep in case["deprecated"] 
                                if isinstance(dep, str))
            
            results.append({
                "Language": language,
                "Category": case["category"],
                "Uses_Deprecated_API": uses_deprecated,
                "Critical_Impact": case["critical_impact"],
                "Generated_Code": response
            })
        
        return pd.DataFrame(results)

def main():
    tester = CriticalCodeTester()
    
    # Test with different models
    models = {
        "GPT-3.5": ChatOpenAI(model="gpt-3.5-turbo"),
        "Claude-3": ChatOpenAI(model="claude-3-haiku-20240307")
    }
    
    all_results = []
    
    for model_name, model in models.items():
        print(f"\nTesting {model_name}...")
        
        # Test Python
        python_results = tester.test_model(model, "python")
        python_results["Model"] = model_name
        
        # Test COBOL
        cobol_results = tester.test_model(model, "cobol")
        cobol_results["Model"] = model_name
        
        all_results.extend([python_results, cobol_results])
    
    # Combine all results
    final_results = pd.concat(all_results, ignore_index=True)
    
    # Analysis
    print("\nDeprecated API Usage Analysis in Critical Operations:")
    print("===================================================")
    
    # Calculate deprecation rates by language and model
    pivot = pd.pivot_table(
        final_results,
        values='Uses_Deprecated_API',
        index=['Model'],
        columns=['Language'],
        aggfunc='mean'
    )
    
    print("\nDeprecation Rates (%):")
    print(pivot * 100)
    
    # Category analysis
    print("\nDeprecation Rates by Category (%):")
    category_stats = final_results.groupby(['Language', 'Category'])['Uses_Deprecated_API'].mean() * 100
    print(category_stats)
    
    # Save detailed results
    final_results.to_csv('critical_code_analysis.csv', index=False)

if __name__ == "__main__":
    main()