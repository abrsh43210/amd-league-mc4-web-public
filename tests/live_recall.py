#!/usr/bin/env python3
"""Real-web retrieval check (needs internet, no GPU): for each question, run the
real search + crawl + parse + rank and report whether the expected answer is in
the context the model would read, and how long gathering took.

    python3 tests/live_recall.py [--min 0.6]
"""
import argparse
import os
import tempfile
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MC4_WORK_DIR", tempfile.mkdtemp(prefix="mc4-live-"))
sys.path.insert(0, str(ROOT / "app"))
from mc4 import web  # noqa: E402
from mc4.pipeline import Session, norm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--min", type=float, default=0.6)
args = ap.parse_args()
qs = json.load(open(ROOT / "tests" / "live_questions.json"))["questions"]

# Raw reachability of each engine endpoint (status, bytes).
for name, url in [("ddg-html", "https://html.duckduckgo.com/html/?q=eiffel+tower"), ("ddg-lite", "https://lite.duckduckgo.com/lite/?q=eiffel+tower"),
                  ("bing", "https://www.bing.com/search?q=eiffel+tower"), ("yahoo", "https://search.yahoo.com/search?p=eiffel+tower"), ("mojeek", "https://www.mojeek.com/search?q=eiffel+tower"),
                  ("wiki-parse", "https://en.wikipedia.org/w/api.php?action=parse&format=json&formatversion=2&prop=text&page=Ras_Dashen")]:
    r = web._get(url, 6)
    print(f"reach {name:10s} {getattr(r, 'status_code', None)} {len(r.content) if r is not None else 0} bytes")

# Which engines answer from this machine at all?
for name, fn in web._ENGINE_FN.items():
    for probe in ("Eiffel Tower height", "Who wrote the novel Things Fall Apart?"):
        t = time.time()
        hs = fn(probe, 6)
        print(f"engine {name:7s} {len(hs):2d} hits {time.time() - t:4.1f}s  [{probe}]  " + " | ".join(h.title[:40] for h in hs[:3]))

LOG = Path(os.environ["MC4_WORK_DIR"]) / "app.log"
hit = 0
for q in qs:
    pos = LOG.stat().st_size if LOG.exists() else 0
    t = time.time()
    s = Session(q["question"], q.get("url"), 25, None)
    s.gather()
    ctx = s.context()
    found = any(norm(a) in norm(" ".join(c.text for c in ctx)) for a in q["answer"])
    hit += found
    print(f"{'OK  ' if found else 'MISS'} {time.time() - t:4.1f}s pages={len(s.pages):2d} ctx={len(ctx):2d}  {q['question']}")
    if not found:
        for h in s.hits[:8]:
            print(f"      hit  {h.engine:6s} {h.title[:50]!r}  {h.url[:90]}")
        for u, p in list(s.pages.items())[:8]:
            print(f"      page {len(p.lines):4d} lines  {p.title[:60]!r}  {u[:100]}")
        if LOG.exists():
            with open(LOG, encoding="utf-8") as fh:
                fh.seek(pos)
                for ln in fh.read().splitlines():
                    if "failed" in ln or "timeout" in ln or " http " in ln:
                        print("      log ", ln[:200])
        for c in ctx[:3]:
            print(f"      ctx {c.score:5.2f} {c.url[:70]}  {c.text[:160]!r}")
rate = hit / len(qs)
print(f"recall {hit}/{len(qs)} = {rate:.0%}")
sys.exit(0 if rate >= args.min else 1)
