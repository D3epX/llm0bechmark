"""
run_all_benchmarks.py
Master orchestrator: runs QnA, Letter generation, and Summary benchmarks
across all models in data/models.csv (or a single model), then aggregates
everything into one CSV for analysis/charting.

Usage:
    python run_all_benchmarks.py --all                  # every model, every task
    python run_all_benchmarks.py --model qwen3.5:4b      # one model, every task
    python run_all_benchmarks.py --all --tasks qna,letter  # every model, selected tasks
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))#syspath.insert(0, str(Path(__file__).parent)) adds the directory containing the current script to the beginning of the Python module search path. This allows the script to import modules from its own directory, even if that directory is not in the default search path. It ensures that local modules can be found and imported correctly during execution.
from benchmark_core import load_models_csv, load_ground_truth, aggregate_results_to_csv
from benchmark_qna import run_qna_benchmark, DATA_PATH as QNA_DATA
from benchmark_letter import run_letter_benchmark, DATA_PATH as LETTER_DATA
from benchmark_summary import run_summary_benchmark, DATA_PATH as SUMMARY_DATA

TASK_RUNNERS = {
    "qna": (run_qna_benchmark, QNA_DATA),
    "letter": (run_letter_benchmark, LETTER_DATA),
    "summary": (run_summary_benchmark, SUMMARY_DATA),
}


def main():
    parser = argparse.ArgumentParser(description="Run all LLM benchmarks")
    parser.add_argument("--model", type=str, help="Single Ollama tag to run, e.g. qwen3.5:4b")
    parser.add_argument("--all", action="store_true", help="Run every model in data/models.csv")
    parser.add_argument("--tasks", type=str, default="qna,letter,summary",
                         help="Comma-separated tasks to run (default: all three)")
    parser.add_argument("--no-metrics", action="store_true", help="Skip ROUGE/BERTScore (timing only — faster)")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]
    compute_quality = not args.no_metrics

    if args.all:
        models = load_models_csv()
        targets = [(m["ollama_tag"], m["model_name"]) for m in models]
    elif args.model:
        targets = [(args.model, args.model)]
    else:
        print("Specify --model <ollama_tag> or --all. See --help.")
        return

    print(f"\nRunning tasks {tasks} across {len(targets)} model(s)...\n")

    for tag, name in targets:
        for task in tasks:
            if task not in TASK_RUNNERS:
                print(f"Unknown task '{task}', skipping.")
                continue
            runner, data_path = TASK_RUNNERS[task]
            dataset = load_ground_truth(data_path)
            try:
                runner(tag, name, dataset, compute_quality)
            except Exception as e:
                print(f"FATAL error running {task} on {name}: {e}")
                print("Continuing with next task/model...")

    print("\nAggregating all results into results/summary_all.csv ...")
    aggregate_results_to_csv("results/qna", "results/qna_summary.csv")
    aggregate_results_to_csv("results/letter", "results/letter_summary.csv")
    aggregate_results_to_csv("results/summary", "results/summary_summary.csv")

    # Combine the three into one master CSV
    import csv as csv_module
    import os
    all_rows = []
    for sub_csv in ["results/qna_summary.csv", "results/letter_summary.csv", "results/summary_summary.csv"]:
        if os.path.exists(sub_csv):
            with open(sub_csv, newline="", encoding="utf-8") as f:
                reader = csv_module.DictReader(f)
                all_rows.extend(list(reader))

    if all_rows:
        with open("results/all_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv_module.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Master CSV -> results/all_results.csv ({len(all_rows)} rows)")

    print("\nDone. Open results/all_results.csv in pandas/Excel for analysis.")


if __name__ == "__main__":
    main()
