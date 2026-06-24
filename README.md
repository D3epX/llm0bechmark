
Benchmarks local LLMs (via Ollama) on three SSPP-relevant administrative tasks:
**QnA**, **Letter Generation**, and **Summarization**.

This benchmark calls Ollama's Python client directly. For raw single-turn generation + timing measurement across many models,this benchmarking harness, direct calls give cleaner timing and simpler debugging.

## On Ground Truth

There is no public ground truth dataset for SSPP-specific French
administrative tasks. The ground truth in `data/*.json` was **authored and validated**
for this project — they are quite relastic for the tasks (leave policy, incident reports, letters, etc.). This is clearly labeled
because reference-based metrics (ROUGE-L, BERTScore) are only as meaningful
as the ground truth they're compared against. For your actual deployment,
replace these with real (anonymized) SSPP examples once available — the
benchmark scripts will work unchanged, just point to new JSON files with
the same structure.

## Project structure

```
llm_benchmark/
├── data/
│   ├── models.csv                  # list of models to benchmark
│   ├── qna_ground_truth.json       # 10 context+query+answer examples
│   ├── letter_ground_truth.json    # 5 instruction+letter examples
│   └── summary_ground_truth.json   # 5 text+summary examples
├── src/
│   ├── benchmark_core.py           # shared engine: Ollama calls, metrics, I/O
│   ├── benchmark_qna.py            # QnA task runner
│   ├── benchmark_letter.py         # Letter generation task runner
│   ├── benchmark_summary.py        # Summarization task runner
│   └── run_all_benchmarks.py       # master orchestrator (all tasks x all models)
├── results/                        # JSON output per (model, task) + aggregated CSVs
├── notebooks/
│   └── llm_benchmark_colab.ipynb   # Google Colab notebook version
└── requirements.txt
```

## Output format

Each run produces a JSON file like `results/qna/qna_Qwen3.5-4B.json`:

```json
{
  "task": "qna",
  "model": "Qwen3.5-4B",
  "ollama_tag": "qwen3.5:4b",
  "timestamp": "2026-06-22 14:30:00",
  "num_examples": 10,
  "examples": [
    {
      "id": "qna_001",
      "query": "...",
      "ground_truth": "...",
      "generated": "...",
      "time_seconds": 4.231,
      "error": null,
      "metrics": {
        "rougeL_f1": 0.62,
        "bertscore_precision": 0.81,
        "bertscore_recall": 0.79,
        "bertscore_f1": 0.80
      }
    }
  ]
}
```

After running, `run_all_benchmarks.py` aggregates everything into
`results/all_results.csv` — one row per (model, task, example), ready for
pandas/Excel/your report charts.

## Usage

### Locally  

```bash
cd llm_benchmark
pip install -r requirements.txt
```
### use conda or venv if you want to isolate dependencies
```bash 
conda create -n LLM_BENCHMARK python=3.14

conda activate LLM_BENCHMARK

pip install -r requirements.txt
```
```bash
# Make sure your models are pulled first:
ollama pull qwen3.5:4b
ollama pull mistral
# ...etc per data/models.csv done 

# Run everything:
python src/run_all_benchmarks.py --all

# Run one model across all tasks:
python src/run_all_benchmarks.py --model qwen3.5:4b

# Run one task only, all models:
python src/run_all_benchmarks.py --all --tasks qna

# Run one task, one model directly:
python src/benchmark_letter.py --model mistral

# Skip quality metrics for a quick timing-only pass:
python src/run_all_benchmarks.py --all --no-metrics
```

### Google Colab

Open `notebooks/llm_benchmark_colab.ipynb` in Colab. It installs Ollama inside
the Colab VM, pulls models, and runs the same benchmark logic in notebook
cells with inline result tables and charts.

## Metrics explained

- **time_seconds**: wall-clock generation time per example (includes model
  "thinking" time if the model has reasoning mode enabled).
- **rougeL_f1**: lexical overlap with ground truth (longest common subsequence).
  Fast, cheap, but penalizes valid paraphrasing.
- **bertscore_f1**: semantic similarity using multilingual BERT embeddings.
  Better correlation with human judgment, especially for French paraphrased
  administrative text where exact wording varies but meaning matches.

both together: a high BERTScore with low ROUGE-L often means the model
produced a valid but differently-worded answer — worth a manual read before
penalizing it.
