#!/usr/bin/env python3
"""End-to-end offline test: real app.py, real search/crawl/parse/rank/grounding code,
against tests/fake_web.py, with the fake LLM (no GPU, no internet).

    python3 tests/run_local.py            # fake LLM (needs requests + beautifulsoup4 + pypdf)
    MC4_REAL_LLM=1 python3 tests/run_local.py   # on an AMD GPU box with weights in /models/llm

Also checks the things that scored zero before: stdout must stay empty, and an
output file must be written for every call, even malformed ones.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app" / "app.py"
QS = json.load(open(ROOT / "tests" / "questions.json"))["questions"]

s = socket.socket(); s.bind(("127.0.0.1", 0)); PORT = s.getsockname()[1]; s.close()
BASE = f"http://127.0.0.1:{PORT}"
web = subprocess.Popen([sys.executable, str(ROOT / "tests" / "fake_web.py"), str(PORT)])
time.sleep(0.8)

work = Path(tempfile.mkdtemp(prefix="mc4-"))
env = dict(os.environ)
env.update({
    "MC4_WORK_DIR": str(work / "work"), "MC4_OUTPUT_DIR": str(work / "output"),
    "MC4_SOCKET": str(work / "model.sock"),
    "MC4_DDG_URL": BASE + "/ddg?q={q}", "MC4_BING_URL": BASE + "/bing?q={q}",
    "MC4_WIKI_API": BASE + "/w/api.php", "MC4_ENGINES": "ddg,bing,wiki",
    "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
    "MC4_DEADLINE_S": env.get("MC4_DEADLINE_S", "20"),
})
if not env.get("MC4_REAL_LLM"):
    oracle = work / "oracle.json"
    oracle.write_text(json.dumps({"questions": [dict(q, question=q["question"].replace("{BASE}", BASE)) for q in QS]}))
    env.update({"MC4_FAKE_LLM": "1", "MC4_FAKE_ORACLE": str(oracle)})

fails = 0


def run(args, label):
    t = time.time()
    r = subprocess.run([sys.executable, str(APP)] + args, env=env, capture_output=True, text=True, timeout=120)
    dt = time.time() - t
    if r.stdout:
        global fails
        fails += 1
        print(f"  !! {label}: stdout not empty: {r.stdout[:200]!r}")
    return r, dt


r, dt = run(["--index"], "warmup")
print(f"warm-up: rc={r.returncode} {dt:.1f}s")

score = 0
for q in QS:
    qtext = q["question"].replace("{BASE}", BASE)
    args = ["--query-id", q["id"], "--query", qtext]
    if q.get("url"):
        args += ["--url", q["url"].replace("{BASE}", BASE)]
    r, dt = run(args, q["id"])
    out = json.load(open(work / "output" / f"{q['id']}_output.json"))
    good = {x.lower() for x in [q["answer"]] + q.get("aliases", [])}
    a_ok = out["answer"].lower() in good if q["answer"] else out["answer"] == ""
    c_ok = (not q["cite"] and out["citations"] == []) or any(c.endswith("/" + q["cite"]) for c in out["citations"])
    ok = a_ok and c_ok and dt < 30
    score += ok
    fails += not ok
    print(f"{q['id']} {'PASS' if ok else 'FAIL'} {dt:4.1f}s  {out}   [{q['what']}]")

# Robustness: odd invocations must still produce a valid file and a clean stdout.
batch = work / "batch.json"
batch.write_text(json.dumps({"queries": [{"id": "b1", "question": QS[0]["question"]}, "What was the population of Aldermoor in the 2020 census?"]}))
single = work / "query_s1.json"
single.write_text(json.dumps({"query": QS[2]["question"]}))
cases = [
    (["--input", str(batch)], ["b1", "query_02"]),
    (["--input", str(single)], ["query_s1"]),
    (["--query-id", "empty", "--query", ""], ["empty"]),
    (["--query-id", "bad_url", "--query", "What is on this page?", "--url", "http://127.0.0.1:1/nothing"], ["bad_url"]),
    (["--input", str(work / "missing.json")], ["missing"]),
    (["--query-id", "weird", "--query", "x", "--unknown-flag", "5"], ["weird"]),
]
for args, stems in cases:
    r, dt = run(args, " ".join(args)[:40])
    for st in stems:
        f = work / "output" / f"{st}_output.json"
        try:
            d = json.load(open(f))
            assert set(d) == {"answer", "citations", "confidence"}
            print(f"robust {st}: ok {dt:.1f}s {d}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"robust {st}: FAIL {e} rc={r.returncode} stderr={r.stderr[-300:]}")

# Grounding check: a number must match a whole number, and the same value written
# another way (date order, percent sign, metre/meter) must still count as found.
sys.path.insert(0, str(ROOT / "app"))
from mc4.pipeline import grounded  # noqa: E402

PAGE = ("The tower opened on March 31, 1889 and is 330 metres tall. Visitors rose 12.5% in 2012 "
        "to 1,200 a day. A ticket costs $12.50 and the lift has Wi-Fi 6, firmware 2.13.0.")
GROUNDING = [
    # false accepts that v2 let through: a short number inside a longer one
    ("12", False), ("20", False), ("13", False), ("2.5", False), ("2.13", False), ("331 metres", False),
    ("1889 and 12", False), ("30 March 1889", False), ("Wi-Fi 60", False),
    # exact values still pass
    ("1889", True), ("2012", True), ("1,200", True), ("1200", True), ("330", True), ("12.5", True),
    ("$12.50", True), ("2.13.0", True), ("WiFi 6", True), ("March 31, 1889", True),
    # false rejects that v2 dropped: same value, different spelling
    ("31 March 1889", True), ("1889-03-31", True), ("31st of March, 1889", True), ("Mar 31 1889", True),
    ("330 meters", True), ("12.5 percent", True), ("12.5 per cent", True), ("12.5 %", True),
    ("March 31, 1889, 330 meters", True),
    ("", False),
]
g_bad = [(a, want) for a, want in GROUNDING if grounded(a, PAGE) != want]
fails += len(g_bad)
print(f"grounding: {len(GROUNDING) - len(g_bad)}/{len(GROUNDING)} ok" + "".join(
    f"\n  !! grounded({a!r}) should be {want}" for a, want in g_bad))

print(f"\nscore {score}/{len(QS)}   failures {fails}   work dir {work}")
subprocess.run(["pkill", "-f", "mc4/server.py"])
web.terminate()
sys.exit(1 if fails else 0)
