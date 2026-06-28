"""
benchmark_core.py

Timer design:
  - time_seconds measures ONLY the Ollama generation call (start → last token)
  - BERTScore / ROUGE computation is always separate, never included in timing
  - compute_metrics_batch() processes all examples in one BERT forward pass (in one batch),
    which is significantly faster than calling one example at a time
    ps: i made a mistake it was scoring after generation, but now i changed it to score in batches.
"""

import json
import time
import csv
import os
from pathlib import Path
from datetime import datetime

import ollama

# Lazy imports — only loaded when scoring is actually requested
_rouge_scorer = None
_bertscore_fn = None


def _load_metric_libs():
    global _rouge_scorer, _bertscore_fn #global variables are used to store the instances of the rouge scorer and bertscore function, so that they can be reused across multiple calls to compute_metrics_batch without having to reload them each time. This improves performance and reduces overhead.
    if _rouge_scorer is None:
        from rouge_score import rouge_scorer as rs
        _rouge_scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
    if _bertscore_fn is None:
        from bert_score import score as bertscore_score
        _bertscore_fn = bertscore_score


def compute_metrics_batch(generated_list: list, gt_list: list, lang: str = "fr") -> list:
    """
    Batch ROUGE-L + BERTScore computation.

    Much faster than one-by-one because BERTScore runs a single BERT forward
    pass over all examples simultaneously instead of N separate passes.

    Args:
        generated_list : list of generated text strings
        gt_list        : list of ground truth strings (same order)
        lang           : language code for BERTScore (default 'fr') most samples are in french, but you can change it to 'en' for english or arabic.

    Returns:
        list of dicts, one per example, with rougeL_f1, bertscore_precision,
        bertscore_recall, bertscore_f1
    """
    _load_metric_libs()

    # ROUGE-L — fast, one by one is fine
    rouge_scores = [
        _rouge_scorer.score(gt, gen)["rougeL"].fmeasure
        for gen, gt in zip(generated_list, gt_list)
    ]

    # BERTScore — expensive; batch call is the key optimization here
    P, R, F1 = _bertscore_fn(
        generated_list,
        gt_list,
        lang=lang,
        model_type="bert-base-multilingual-cased",
        verbose=False,
        # batch_size=8 by default — reduce to 4 if you hit OOM during scoring
    )

    return [
        {
            "rougeL_f1": round(float(rouge_scores[i]), 4),
            "bertscore_precision": round(float(P[i]), 4),
            "bertscore_recall": round(float(R[i]), 4),
            "bertscore_f1": round(float(F1[i]), 4),
        }
        for i in range(len(generated_list))
    ]


def call_ollama(model_tag: str, prompt: str, system: str = None) -> dict:
    """
    Call a local Ollama model and measure ONLY generation time.

    The timer starts immediately before the API call and stops immediately
    after the last token is received. No metric computation, no file I/O,
    no post-processing is included in the timing.

    Returns:
        dict with keys:
            content      : generated text (empty string on error)
            time_seconds : wall-clock generation time in seconds
            error        : error message string, or None on success
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    start = time.perf_counter()
    try:
        response = ollama.chat(
            model=model_tag,
            messages=messages,
            options={"temperature": 0.3},#lower temperature means less randomness and more deterministic output, which is often desirable for benchmarking and evaluation purposes. It helps to reduce variability in the generated responses, making it easier to compare results across different runs or models.
        )
        elapsed = time.perf_counter() - start   # ← timer stops here
        return {
            "content": response["message"]["content"],
            "time_seconds": round(elapsed, 3),
            "error": None,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "content": "",
            "time_seconds": round(elapsed, 3),
            "error": str(e),
        }


def load_models_csv(path: str = "data/models.csv") -> list:
    """Load the list of models to benchmark from CSV."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_ground_truth(path: str) -> list:
    """Load a ground truth JSON dataset."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_result(result: dict, output_dir: str, task_name: str, model_name: str) -> str:
    """Save a run result dict as a JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe_model = model_name.replace(":", "_").replace("/", "_")
    fpath = os.path.join(output_dir, f"{task_name}_{safe_model}.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return fpath


def aggregate_results_to_csv(results_dir: str, output_csv: str):
    """
    Scan all JSON result files in results_dir and flatten into one CSV,
    one row per (model, task, example).
    """
    rows = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(results_dir, fname), encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("examples", []):
            rows.append({
                "task":             data.get("task"),
                "model":            data.get("model"),
                "example_id":       item.get("id"),
                "time_seconds":     item.get("time_seconds"),
                "error":            item.get("error") or "",
                "rougeL_f1":        item.get("metrics", {}).get("rougeL_f1", ""),
                "bertscore_f1":     item.get("metrics", {}).get("bertscore_f1", ""),
                "generated_preview":(item.get("generated", "") or "")[:120].replace("\n", " "),
            })
    if not rows:
        print(f"  No results found in {results_dir}")
        return
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Aggregated {len(rows)} rows → {output_csv}")


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S") #backup timestamp for the results, so that we can keep track of when the benchmark was run. This is useful for comparing results over time, especially if the models or datasets are updated.
