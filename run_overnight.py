"""
run_overnight.py
Simple queue: runs QnA, Letter, Summary for each model one at a time.
Waits between each run so the CPU can recover.

Models: gemma4:e4b-it-q4_K_M | gemma4:e2b | phi4-mini

Just copy this file into your llm_benchmark folder and run:
    python run_overnight.py
"""

import subprocess
import time
import sys
from datetime import datetime


# ── Edit these if needed ──────────────────────────────────────────────────────

MODELS = [
    "qwen3.5:4b"
]

TASKS = [
    "src/benchmark_qna.py",
    "src/benchmark_letter.py"
]

PAUSE_SECONDS = 180   # 3 minutes between each run — let CPU breathe

# ─────────────────────────────────────────────────────────────────────────────


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open("overnight_log.txt", "a") as f:
        f.write(line + "\n")


def pause(seconds):
    log(f"Waiting {seconds}s for CPU to cool down...")
    for remaining in range(seconds, 0, -30):
        time.sleep(min(30, remaining))
        if remaining > 30:
            log(f"  {remaining - 30}s left...")
    log("Done waiting. Starting next run.")


def run(model, task_script):
    log(f"START — model={model}  task={task_script}")
    start = time.time()

    result = subprocess.run(
        [sys.executable, task_script, "--model", model, "--no-metrics"],
    )

    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else "FAILED"
    log(f"{status} — {model} / {task_script} — {elapsed:.0f}s")
    return result.returncode == 0


def main():
    total = len(MODELS) * len(TASKS)
    job = 0

    log("=" * 50)
    log("OVERNIGHT BENCHMARK QUEUE")
    log(f"Models: {MODELS}")
    log(f"Tasks:  {TASKS}")
    log(f"Total jobs: {total}")
    log("=" * 50)

    for model in MODELS:
        for task in TASKS:
            job += 1
            log(f"\nJob {job}/{total}")
            run(model, task)

            # Pause after every job except the last one
            if job < total:
                pause(PAUSE_SECONDS)

    log("=" * 50)
    log("ALL DONE. Check results/ folder for output files.")
    log("=" * 50)


if __name__ == "__main__":
    main()
