"""
benchmark_letter.py


Flow per model:
  Phase 1 — Generation
    letter 1 → generate → save raw result
    letter 2 → generate → save raw result
    ...all letters done, timing recorded for each
  Phase 2 — Scoring (batch, after ALL generation is complete)
    all generated letters scored together in one BERTScore pass
    results file updated with metrics

Usage:
    python benchmark_letter.py --model mistral
    python benchmark_letter.py --all
    python benchmark_letter.py --all --no-metrics
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))#syspath.insert(0, str(Path(__file__).parent)) adds the directory containing the current script to the beginning of the Python module search path. This allows the script to import modules from its own directory, even if that directory is not in the default search path. It ensures that local modules can be found and imported correctly during execution.
from benchmark_core import (
    call_ollama, compute_metrics_batch, load_models_csv,
    load_ground_truth, save_result, timestamp
)

SYSTEM_PROMPT = (
    "Tu es un assistant administratif expert en rédaction de courriers officiels "
    "pour une entreprise pétrolière algérienne (subsidiaire SSPP / Sonatrach). "
    "Tu rédiges uniquement en français administratif formel. Respecte la structure "
    "imposée : en-tête, date, objet, formule d'appel, corps structuré, formule de "
    "politesse finale, signature. Ne génère rien en dehors de la lettre elle-même."
)

DATA_PATH   = "data/letter_ground_truth.json"
RESULTS_DIR = "results/letter"
TASK_NAME   = "letter"


def run_letter_benchmark(model_tag: str, model_name: str, dataset: list,
                         compute_quality_metrics: bool = True):
    print(f"\n{'='*60}")
    print(f"Letter Generation Benchmark — {model_name} ({model_tag})")
    print(f"{'='*60}")

    examples = []

    # ── Phase 1: generate every letter, record timing ────────────────────────
    print("\n  Phase 1 — Generation")
    for item in dataset:
        print(f"    [{item['id']}] generating...", end=" ", flush=True)

        result = call_ollama(model_tag, item["query"], system=SYSTEM_PROMPT)

        if result["error"]:
            print(f"ERROR — {result['error']}")
        else:
            print(f"{result['time_seconds']}s")

        examples.append({
            "id":           item["id"],
            "query":        item["query"],
            "ground_truth": item["ground_truth"],
            "generated":    result["content"],
            "time_seconds": result["time_seconds"],   # pure generation time only
            "error":        result["error"],
        })

    # ── Save after generation, before scoring ────────────────────────────────
    output = {
        "task":         TASK_NAME,
        "model":        model_name,
        "ollama_tag":   model_tag,
        "timestamp":    timestamp(),
        "num_examples": len(examples),
        "examples":     examples,
    }
    path = save_result(output, RESULTS_DIR, TASK_NAME, model_name)
    print(f"\n  Generation complete. Saved (no metrics yet) → {path}")

    # ── Phase 2: batch scoring after ALL generation is done ──────────────────
    if compute_quality_metrics:
        print("\n  Phase 2 — Batch scoring (ROUGE-L + BERTScore)")
        print("  (This runs once for all letters together — faster than one-by-one)")

        valid_indices = [
            i for i, ex in enumerate(examples)
            if not ex["error"] and ex["generated"]
        ]

        if valid_indices:
            generated_texts = [examples[i]["generated"] for i in valid_indices]
            gt_texts        = [examples[i]["ground_truth"] for i in valid_indices]

            metrics_list = compute_metrics_batch(generated_texts, gt_texts)

            for idx, metrics in zip(valid_indices, metrics_list):
                examples[idx]["metrics"] = metrics
                print(f"    [{examples[idx]['id']}] "
                      f"ROUGE-L={metrics['rougeL_f1']:.3f}  "
                      f"BERTScore={metrics['bertscore_f1']:.3f}")
        else:
            print("  No valid generations to score.")

        output["examples"] = examples
        save_result(output, RESULTS_DIR, TASK_NAME, model_name)
        print(f"\n  Scoring complete. File updated → {path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLMs on letter generation task")
    parser.add_argument("--model", type=str,
                        help="Ollama tag of a single model, e.g. mistral")
    parser.add_argument("--all", action="store_true",
                        help="Run every model listed in data/models.csv")
    parser.add_argument("--no-metrics", action="store_true",
                        help="Skip ROUGE/BERTScore — generation timing only")
    args = parser.parse_args()

    dataset = load_ground_truth(DATA_PATH)
    compute = not args.no_metrics

    if args.all:
        for m in load_models_csv():
            run_letter_benchmark(m["ollama_tag"], m["model_name"], dataset, compute)
    elif args.model:
        run_letter_benchmark(args.model, args.model, dataset, compute)
    else:
        print("Specify --model <tag> or --all.  Use --help for options.")


if __name__ == "__main__":
    main()
