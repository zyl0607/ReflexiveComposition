# case_studies/code_security/evaluate.py
"""
Evaluation framework for the code security case study.

This script provides automated evaluation capabilities for measuring
the effectiveness of Reflexive Composition in reducing historical bias
in code generation, specifically targeting security-related issues.
"""

import os
import re
import json
import logging
import argparse
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CodeSecurityEvaluator:
    """
    Framework for evaluating security aspects of generated code.
    
    This evaluator compares baseline and KG-enhanced code generation
    in terms of security patterns, API usage, and vulnerability avoidance.
    """
    
    def __init__(self, 
                 security_patterns_path: str,
                 output_dir: str = "evaluation_results"):
        """
        Initialize the code security evaluator.
        
        Args:
            security_patterns_path: Path to security patterns file
            output_dir: Directory to save evaluation results
        """
        self.patterns = self._load_patterns(security_patterns_path)
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = {
            "baseline": [],
            "kg_enhanced": []
        }
    
    def _load_patterns(self, patterns_path: str) -> Dict[str, Any]:
        """
        Load security patterns from a JSON file.
        
        Args:
            patterns_path: Path to patterns file
            
        Returns:
            Dictionary of security patterns
        """
        try:
            with open(patterns_path, 'r') as f:
                patterns = json.load(f)
            logger.info(f"Loaded {len(patterns.get('deprecations', []))} deprecated APIs")
            self.patterns = {
                "deprecated_apis": patterns.get("deprecations", []),
                "vulnerabilities": patterns.get("vulnerabilities", []),
                "secure_patterns": patterns.get("secure_patterns", [])
            }
            return patterns
        except Exception as e:
            logger.error(f"Error loading security patterns: {e}")
            return {
                "deprecations": [],
                "secure_alternatives": {},
                "vulnerability_patterns": []
            }
    
    def evaluate_code(self, 
                      code: str, 
                      task_prompt: str, 
                      system_type: str) -> Dict[str, Any]:
        """
        Evaluate the security of a code snippet.
        
        Args:
            code: Generated code snippet
            task_prompt: Original task prompt
            system_type: "baseline" or "kg_enhanced"
            
        Returns:
            Evaluation results
        """
        # Initialize results
        results = {
            "task_prompt": task_prompt,
            "system_type": system_type,
            "code_length": len(code),
            "deprecated_apis_found": [],
            "secure_alternatives_used": [],
            "vulnerabilities_found": [],
            "security_score": 0.0
        }
        
        # Check for deprecated APIs
        for api in self.patterns.get("deprecations", []):
            if re.search(rf'\b{re.escape(api)}\b', code):
                results["deprecated_apis_found"].append(api)
        
        # Check for secure alternatives
        for api, alternatives in self.patterns.get("secure_patterns", {}).items():
            for alt in alternatives:
                if re.search(rf'\b{re.escape(alt)}\b', code):
                    results["secure_alternatives_used"].append(alt)
        
        # Check for vulnerability patterns
        for pattern in self.patterns.get("vulnerabilities", []):
            pattern_regex = pattern.get("regex")
            if pattern_regex and re.search(pattern_regex, code):
                results["vulnerabilities_found"].append(pattern.get("name", "Unknown"))
        
        # Calculate security score
        num_deprecated = len(results["deprecated_apis_found"])
        num_secure = len(results["secure_alternatives_used"])
        num_vulns = len(results["vulnerabilities_found"])
        
        # Simple scoring formula: penalize deprecated APIs and vulnerabilities, reward secure alternatives
        results["security_score"] = max(0.0, min(1.0, 0.5 - 0.2 * num_deprecated + 0.2 * num_secure - 0.2 * num_vulns))
        
        # Store results
        self.results[system_type].append(results)
        
        return results
    
    def add_result(self, 
                  code: str, 
                  task_prompt: str, 
                  test_case: Dict[str, Any],
                  system_type: str) -> Dict[str, Any]:
        """
        Add an evaluation result with additional test case information.
        
        Args:
            code: Generated code snippet
            task_prompt: Original task prompt
            test_case: Test case dictionary with ground truth info
            system_type: "baseline" or "kg_enhanced"
            
        Returns:
            Evaluation results
        """
        # Get base evaluation
        results = self.evaluate_code(code, task_prompt, system_type)
        
        # Add test case specific information
        results["expected_api"] = test_case.get("deprecated_api", "")
        results["expected_alternative"] = test_case.get("secure_alternative", "")
        results["risk_category"] = test_case.get("risk", "")
        
        # Calculate precise scores
        results["deprecated_api_avoided"] = results["expected_api"] not in results["deprecated_apis_found"]
        results["secure_alternative_used"] = any(alt in code for alt in results["expected_alternative"].split(" or "))
        
        return results
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze evaluation results and generate statistics.
        
        Returns:
            Dictionary of analysis results
        """
        analysis = {
            "baseline": {
                "sample_count": len(self.results["baseline"]),
                "deprecated_api_usage_rate": 0.0,
                "secure_alternative_usage_rate": 0.0,
                "vulnerability_detection_rate": 0.0,
                "avg_security_score": 0.0
            },
            "kg_enhanced": {
                "sample_count": len(self.results["kg_enhanced"]),
                "deprecated_api_usage_rate": 0.0,
                "secure_alternative_usage_rate": 0.0,
                "vulnerability_detection_rate": 0.0,
                "avg_security_score": 0.0
            },
            "improvement": {
                "deprecated_api_reduction": 0.0,
                "secure_alternative_increase": 0.0,
                "vulnerability_reduction": 0.0,
                "security_score_improvement": 0.0
            }
        }
        
        # Calculate baseline metrics
        if self.results["baseline"]:
            baseline = self.results["baseline"]
            baseline_deprecated_rate = sum(1 for r in baseline if r["deprecated_apis_found"]) / len(baseline)
            baseline_secure_rate = sum(1 for r in baseline if r["secure_alternatives_used"]) / len(baseline)
            baseline_vuln_rate = sum(1 for r in baseline if r["vulnerabilities_found"]) / len(baseline)
            baseline_avg_score = sum(r["security_score"] for r in baseline) / len(baseline)
            
            analysis["baseline"].update({
                "deprecated_api_usage_rate": baseline_deprecated_rate,
                "secure_alternative_usage_rate": baseline_secure_rate,
                "vulnerability_detection_rate": baseline_vuln_rate,
                "avg_security_score": baseline_avg_score
            })
        
        # Calculate KG-enhanced metrics
        if self.results["kg_enhanced"]:
            enhanced = self.results["kg_enhanced"]
            enhanced_deprecated_rate = sum(1 for r in enhanced if r["deprecated_apis_found"]) / len(enhanced)
            enhanced_secure_rate = sum(1 for r in enhanced if r["secure_alternatives_used"]) / len(enhanced)
            enhanced_vuln_rate = sum(1 for r in enhanced if r["vulnerabilities_found"]) / len(enhanced)
            enhanced_avg_score = sum(r["security_score"] for r in enhanced) / len(enhanced)
            
            analysis["kg_enhanced"].update({
                "deprecated_api_usage_rate": enhanced_deprecated_rate,
                "secure_alternative_usage_rate": enhanced_secure_rate,
                "vulnerability_detection_rate": enhanced_vuln_rate,
                "avg_security_score": enhanced_avg_score
            })
        
        # Calculate improvement
        if self.results["baseline"] and self.results["kg_enhanced"]:
            baseline = analysis["baseline"]
            enhanced = analysis["kg_enhanced"]
            
            # Calculate relative improvement
            deprecated_reduction = (
                (baseline["deprecated_api_usage_rate"] - enhanced["deprecated_api_usage_rate"]) / 
                max(0.001, baseline["deprecated_api_usage_rate"])
            ) * 100
            
            secure_increase = (
                (enhanced["secure_alternative_usage_rate"] - baseline["secure_alternative_usage_rate"]) / 
                max(0.001, baseline["secure_alternative_usage_rate"])
            ) * 100
            
            vuln_reduction = (
                (baseline["vulnerability_detection_rate"] - enhanced["vulnerability_detection_rate"]) / 
                max(0.001, baseline["vulnerability_detection_rate"])
            ) * 100
            
            score_improvement = (
                (enhanced["avg_security_score"] - baseline["avg_security_score"]) / 
                max(0.001, baseline["avg_security_score"])
            ) * 100
            
            analysis["improvement"].update({
                "deprecated_api_reduction": deprecated_reduction,
                "secure_alternative_increase": secure_increase,
                "vulnerability_reduction": vuln_reduction,
                "security_score_improvement": score_improvement
            })
        
        return analysis
    
    def generate_report(self) -> None:
        """
        Generate evaluation report and save results.
        """
        # Analyze results
        analysis = self.analyze_results()
        
        # Save detailed results to JSON
        results_path = os.path.join(self.output_dir, "detailed_results.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save analysis to JSON
        analysis_path = os.path.join(self.output_dir, "analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Create dataframe for CSV export
        all_results = []
        for system_type in ["baseline", "kg_enhanced"]:
            for result in self.results[system_type]:
                all_results.append({
                    "system_type": system_type,
                    "task_prompt": result["task_prompt"],
                    "deprecated_apis_count": len(result["deprecated_apis_found"]),
                    "secure_alternatives_count": len(result["secure_alternatives_used"]),
                    "vulnerabilities_count": len(result["vulnerabilities_found"]),
                    "security_score": result["security_score"],
                    "deprecated_api_avoided": result.get("deprecated_api_avoided", False),
                    "secure_alternative_used": result.get("secure_alternative_used", False)
                })
        
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(self.output_dir, "results.csv"), index=False)
        
        # Print summary
        print("\n=== Security Evaluation Summary ===")
        print(f"Total samples: Baseline ({analysis['baseline']['sample_count']}), "
              f"KG-Enhanced ({analysis['kg_enhanced']['sample_count']})")
        
        print("\nMetrics:")
        print(f"Deprecated API Usage: "
              f"{analysis['baseline']['deprecated_api_usage_rate']*100:.1f}% → "
              f"{analysis['kg_enhanced']['deprecated_api_usage_rate']*100:.1f}% "
              f"({analysis['improvement']['deprecated_api_reduction']:.1f}% reduction)")
        
        print(f"Secure Alternative Usage: "
              f"{analysis['baseline']['secure_alternative_usage_rate']*100:.1f}% → "
              f"{analysis['kg_enhanced']['secure_alternative_usage_rate']*100:.1f}% "
              f"({analysis['improvement']['secure_alternative_increase']:.1f}% increase)")
        
        print(f"Security Score: "
              f"{analysis['baseline']['avg_security_score']:.2f} → "
              f"{analysis['kg_enhanced']['avg_security_score']:.2f} "
              f"({analysis['improvement']['security_score_improvement']:.1f}% improvement)")
        
        logger.info(f"Evaluation report saved to {self.output_dir}")

def create_example_patterns():
    """
    Create example security patterns file for evaluation.
    """
    patterns = {
        "deprecated_apis": [
            "datetime.utcnow",
            "urllib.urlopen",
            "random.random",
            "hashlib.md5",
            "threading.activeCount",
            "pickle.loads",
            "ast.Num"
        ],
        "secure_alternatives": {
            "datetime.utcnow": ["datetime.now(tz=datetime.UTC)"],
            "urllib.urlopen": ["requests.get", "requests.get()"],
            "random.random": ["secrets.token_bytes", "secrets.token_hex"],
            "hashlib.md5": ["hashlib.sha256", "hashlib.blake2b"],
            "threading.activeCount": ["threading.active_count"],
            "pickle.loads": ["json.loads", "marshal.loads"]
        },
        "vulnerability_patterns": [
            {
                "name": "SQL Injection",
                "regex": r"execute\s*\(\s*[\'\"][^\'\"]*\%s",
                "description": "SQL query without proper parameterization"
            },
            {
                "name": "Shell Injection",
                "regex": r"subprocess\.call\s*\(\s*.*shell\s*=\s*True",
                "description": "Subprocess call with shell=True"
            },
            {
                "name": "Insecure Hash",
                "regex": r"hashlib\.md5\s*\(",
                "description": "Usage of insecure MD5 hash algorithm"
            },
            {
                "name": "Hardcoded Credentials",
                "regex": r"password\s*=\s*[\'\"][^\'\"\s]{3,}[\'\"]",
                "description": "Hardcoded password in code"
            }
        ]
    }
    
    # Create directory if it doesn't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Save patterns to file
    patterns_path = os.path.join(data_dir, "security_patterns.json")
    with open(patterns_path, 'w') as f:
        json.dump(patterns, f, indent=2)
    
    logger.info(f"Created example security patterns at {patterns_path}")
    return patterns_path

def main():
    """Run the evaluation framework."""
    parser = argparse.ArgumentParser(description="Evaluate code security using Reflexive Composition")
    parser.add_argument("--patterns", type=str, help="Path to security patterns file")
    parser.add_argument("--results", type=str, help="Path to code generation results file")
    parser.add_argument("--output", type=str, default="evaluation_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Create example patterns if not provided
    patterns_path = args.patterns or create_example_patterns()
    
    # Initialize evaluator
    evaluator = CodeSecurityEvaluator(patterns_path, args.output)
    
    # If results file is provided, evaluate from file
    if args.results:
        with open(args.results, 'r') as f:
            results = json.load(f)
        
        for result in results.get("baseline", []):
            evaluator.evaluate_code(
                result["code"], 
                result["task_prompt"], 
                "baseline"
            )
        
        for result in results.get("kg_enhanced", []):
            evaluator.evaluate_code(
                result["code"], 
                result["task_prompt"], 
                "kg_enhanced"
            )
    else:
        # Example evaluation (for demonstration)
        print("No results file provided. Running with example data...")
        
        # Example baseline code with security issues
        baseline_code = """
def get_current_utc_time():
    from datetime import datetime
    return datetime.utcnow()

def generate_token(length=32):
    import random
    import string
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def fetch_data(url):
    import urllib
    response = urllib.urlopen(url)
    return response.read()
        """
        
        # Example KG-enhanced code with security improvements
        enhanced_code = """
def get_current_utc_time():
    from datetime import datetime
    return datetime.now(tz=datetime.UTC)

def generate_token(length=32):
    import secrets
    return secrets.token_hex(length // 2)  # Each byte becomes 2 hex chars

def fetch_data(url):
    import requests
    response = requests.get(url, verify=True)
    return response.content
        """
        
        # Evaluate example code
        evaluator.evaluate_code(baseline_code, "Implement secure utility functions", "baseline")
        evaluator.evaluate_code(enhanced_code, "Implement secure utility functions", "kg_enhanced")
    
    # Generate report
    evaluator.generate_report()

if __name__ == "__main__":
    main()
