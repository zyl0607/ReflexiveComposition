
import os
import subprocess
import argparse
from reflexive_composition.eval import metrics

# Define use case runners
USE_CASES = {
    "code_security": "examples/code_security/run_codegen.py",
    "temporal_qa": "examples/temporal_qa/run_temporal.py",
    "medical_data": "examples/medical_data/run_medical.py"
}

# Define output paths for post-analysis
OUTPUT_LOGS = {
    "code_security": "outputs/code_security_eval.jsonl",
    "temporal_qa": "outputs/temporal_eval.jsonl",
    "medical_data": "outputs/medical_eval.jsonl"
}

def run_use_case(label, script_path, interactive=True):
    args = ["python", script_path]
    if not interactive:
        args.append("--no-hitl")
    print(f"Running: {label} with arguments: {args}")

    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(f"[{label}] {line}", end="")
    process.wait()

def evaluate_outputs():
    for label, output_path in OUTPUT_LOGS.items():
        if not os.path.exists(output_path):
            print(f"[{label}] No output log found at: {output_path}")
            continue
        print(f"[{label}] Evaluating output at: {output_path}")
        cases = metrics.load_jsonl(output_path)
        if "code" in label:
            results = metrics.evaluate_hitl_intervention(cases)
        elif "temporal" in label or "medical" in label:
            gold = [c["gold_answer"] for c in cases if "gold_answer" in c]
            pred = [c["generated_answer"] for c in cases if "generated_answer" in c]
            results = metrics.evaluate_generation_accuracy(gold, pred)
        else:
            results = {}
        print(f"[{label}] Metrics: {results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="Enable manual HITL interaction during runs")
    args = parser.parse_args()

    interactive = args.interactive

    for label, script in USE_CASES.items():
        run_use_case(label, script, interactive=interactive)
    evaluate_outputs()
