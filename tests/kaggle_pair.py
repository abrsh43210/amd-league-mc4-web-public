#!/usr/bin/env python3
"""Run two checkouts of this repo side by side on the same questions at the same time
(one GPU each) and print a per-question comparison. Used on Kaggle's free 2xT4.

    python tests/kaggle_pair.py --a /kaggle/working/v5 --b /kaggle/working/v6
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--work", default="/kaggle/working/pair")
ap.add_argument("--sets", default="hard_questions,live_questions")
args = ap.parse_args()

HERE = Path(__file__).resolve().parent
work = Path(args.work)
work.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pypdf", "truststore", "beautifulsoup4"], check=False)

from huggingface_hub import snapshot_download  # noqa: E402

model = snapshot_download("Qwen/Qwen3-4B-Instruct-2507", revision="cdbee75f17c01a7cc42f958dc650907174af0554",
                          local_dir=str(work / "model"))
print("model at", model, flush=True)

for name in args.sets.split(","):
    procs = []
    for gpu, (tag, root) in enumerate((("a", args.a), ("b", args.b))):
        for f in ("gpu_eval.py", f"{name}.json"):
            if not (Path(root) / "tests" / f).exists():
                shutil.copy(HERE / f, Path(root) / "tests" / f)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MC4_MODEL_DIR=model,
                   MC4_WORK_DIR=str(work / f"work-{tag}-{name}"))
        log = open(work / f"{name}-{tag}.log", "w")
        procs.append(subprocess.Popen(
            [sys.executable, "tests/gpu_eval.py", "--questions", f"tests/{name}.json",
             "--out", str(work / f"{name}-{tag}.json")], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    res = {}
    for tag in ("a", "b"):
        f = work / f"{name}-{tag}.json"
        if f.exists():
            res[tag] = json.load(open(f))
        else:
            print(f"== {name} {tag}: no result; log tail:\n" + open(work / f"{name}-{tag}.log").read()[-1500:], flush=True)
    for tag, r in res.items():
        print(f"== {name} {tag} SUMMARY {json.dumps(r['summary'])}", flush=True)
    if len(res) == 2:
        for ra, rb in zip(res["a"]["rows"], res["b"]["rows"]):
            mark = "same" if ra["ok"] == rb["ok"] else ("B-BETTER" if rb["ok"] else "A-BETTER")
            print(f"{mark:8} a={'OK' if ra['ok'] else 'no'} {ra['s']:5.1f}s {ra['answer'][:28]!r:32} "
                  f"b={'OK' if rb['ok'] else 'no'} {rb['s']:5.1f}s c={rb['conf']:.1f} {rb['answer'][:28]!r:32} | {ra['q'][:60]}", flush=True)
    for tag in ("a", "b"):
        lg = work / f"work-{tag}-{name}" / "app.log"
        if lg.exists():
            t = lg.read_text(errors="replace")
            print(f"== {name} {tag} log: ddg0={t.count('engine ddg -> 0 hits')} bing0={t.count('engine bing -> 0 hits')} "
                  f"yahoo0={t.count('engine yahoo -> 0 hits')} wiki0={t.count('engine wiki -> 0 hits')} "
                  f"mojeek0={t.count('engine mojeek -> 0 hits')} wiki429={t.count('http 429')} "
                  f"searches={t.count('search: ')} rested={t.count('resting it')} unverified={t.count('set aside')}", flush=True)
