#!/usr/bin/env python3
"""End-to-end check with the REAL model on a GPU and the REAL web.

For each question: run the whole pipeline (search, fetch, rank, one model call,
grounding) and report the answer, whether it contains an expected answer, the
seconds used, and peak GPU memory. Nothing here is used by the submission image.

    MC4_MODEL_DIR=/path/to/Qwen3-4B-Instruct-2507 python3 tests/gpu_eval.py [--limit N] [--out results.json]
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MC4_WORK_DIR", tempfile.mkdtemp(prefix="mc4-gpu-"))
sys.path.insert(0, str(ROOT / "app"))

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default="")
ap.add_argument("--questions", default=str(ROOT / "tests" / "live_questions.json"))
args = ap.parse_args()

import torch  # noqa: E402

from mc4 import paths  # noqa: E402
from mc4.model import LocalLLM  # noqa: E402
from mc4.pipeline import answer, norm  # noqa: E402

qs = json.load(open(args.questions))["questions"]
if args.limit:
    qs = qs[: args.limit]

gpu = torch.cuda.is_available()
print(f"torch {torch.__version__}  cuda {gpu}  device {torch.cuda.get_device_name(0) if gpu else 'cpu'}", flush=True)
t0 = time.time()
llm = LocalLLM(paths.MODEL_DIR)
load_s = time.time() - t0
print(f"model loaded in {load_s:.1f}s  vram {torch.cuda.memory_allocated() / 2**30 if gpu else 0:.2f} GiB", flush=True)

rows, ok, empty = [], 0, 0
for q in qs:
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    try:
        a, cites, conf = answer(q["question"], q.get("url"), paths.DEADLINE_S, llm)
        err = ""
    except Exception as e:  # noqa: BLE001
        a, cites, conf, err = "", [], 0.0, f"{type(e).__name__}: {e}"
    dt = time.time() - t
    good = bool(a) and any(norm(x) in norm(a) for x in q["answer"])
    ok += good
    empty += not a
    peak = torch.cuda.max_memory_allocated() / 2**30 if gpu else 0.0
    rows.append({"q": q["question"], "answer": a, "expected": q["answer"], "ok": good, "s": round(dt, 1),
                 "conf": conf, "cites": cites[:3], "peak_gib": round(peak, 2), "err": err})
    print(f"{'OK   ' if good else ('EMPTY' if not a else 'WRONG')} {dt:5.1f}s conf={conf:.2f} peak={peak:.1f}G  {q['question']}  ->  {a[:80]!r}  {err}", flush=True)

times = sorted(r["s"] for r in rows)
summary = {"n": len(rows), "correct": ok, "empty": empty, "wrong": len(rows) - ok - empty, "load_s": round(load_s, 1),
           "median_s": times[len(times) // 2] if times else 0, "max_s": times[-1] if times else 0,
           "peak_gib": max((r["peak_gib"] for r in rows), default=0), "deadline_s": paths.DEADLINE_S}
print("SUMMARY " + json.dumps(summary), flush=True)
if args.out:
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=1)
