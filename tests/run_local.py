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

# A stalling search engine must not hold the whole search (it used to take 3x the
# timeout, which left no time to read pages). Stub the HTTP layer: no network here.
from mc4 import web as _web  # noqa: E402


class _WikiOK:
    status_code, text = 200, ""

    def json(self):
        return {"query": {"search": [{"title": "Eiffel Tower", "snippet": "opened in 1889"}]}}


def _stub(stall):
    def fake_get(url, timeout, method="GET", **kw):
        if stall(url):
            time.sleep(max(0.5, timeout))             # behaves like the real wall-clock cap
            return None
        return _WikiOK() if "api.php" in url else None
    return fake_get


_real_get, T = _web._get, 2.0
for label, stall, need_hits in [("every engine stalls", lambda u: True, False),
                                ("only DuckDuckGo stalls", lambda u: "duckduckgo" in u, True)]:
    _web._get = _stub(stall)
    t0 = time.time()
    hits = _web.search(["When did the Eiffel Tower open?"], timeout=T)
    took = time.time() - t0
    ok = took <= T + 1.5 and (bool(hits) or not need_hits)
    fails += not ok
    print(f"stall ({label}): {'ok' if ok else 'FAIL'} search returned in {took:.1f}s (limit {T + 1.5:.1f}s), {len(hits)} hits")
_web._get = _real_get

# Throttled engines (v5). Stubbed HTTP again: each route is checked on its own.
class _R:
    def __init__(self, status, text="", data=None, headers=None):
        self.status_code, self.text, self._d, self.headers = status, text, data, headers or {}

    def json(self):
        if self._d is None:
            raise ValueError("no json")
        return self._d


def _throttle_case(label, fake, fn, want_urls, limit):
    global fails
    calls = []

    def g(url, timeout, method="GET", **kw):
        calls.append(url)
        return fake(url, len(calls))
    _web._get = g
    t0 = time.time()
    got = [h.url for h in fn("Eiffel Tower", 3.0)]
    took = time.time() - t0
    ok = got == want_urls and took <= limit
    fails += not ok
    print(f"throttle ({label}): {'ok' if ok else 'FAIL'} {len(calls)} requests, {took:.1f}s, {got}")


_API = {"query": {"search": [{"title": "Eiffel Tower", "snippet": "opened in <b>1889</b>"}]}}
_REST = {"pages": [{"key": "Eiffel_Tower", "title": "Eiffel Tower", "excerpt": "opened in <span>1889</span>"}]}
_WURL = [_web.WIKI_API.rsplit("/w/api.php", 1)[0] + "/wiki/Eiffel_Tower"]
_WHTML = ('<ul><li class="mw-search-result"><div class="mw-search-result-heading"><a href="/wiki/Eiffel_Tower" '
          'title="Eiffel Tower">Eiffel Tower</a></div><div class="searchresult">opened in 1889</div></li></ul>')
_throttle_case("Wikipedia API 429, REST route answers",
               lambda u, n: _R(200, data=_REST) if "rest.php" in u else _R(429, headers={"retry-after": "20"}),
               _web._wiki, _WURL, 1.0)
_throttle_case("Wikipedia all routes 429, short Retry-After honoured",
               lambda u, n: _R(200, data=_API) if n == 4 else _R(429, headers={"retry-after": "1"}),
               _web._wiki, _WURL, 2.0)
_throttle_case("Wikipedia all routes 429, long Retry-After not waited for",
               lambda u, n: _R(429, headers={"retry-after": "20"}), _web._wiki, [], 1.0)
_throttle_case("Wikipedia API answers garbage, REST route answers",
               lambda u, n: _R(200, data=_REST) if "rest.php" in u else _R(200, text="<html>"),
               _web._wiki, _WURL, 1.0)
_throttle_case("Yahoo 500 then 200: one retry",
               lambda u, n: _R(500) if n == 1 else _R(200, text='<div class="algo"><h3><a href="https://example.org/y">Y</a></h3></div>'),
               _web._yahoo, ["https://example.org/y"], 1.5)
_throttle_case("Yahoo 500 twice: gives up after one retry",
               lambda u, n: _R(500), _web._yahoo, [], 1.5)
_throttle_case("Wikipedia API and REST 429, search page answers",
               lambda u, n: _R(200, text=_WHTML) if "index.php" in u else _R(429, headers={"retry-after": "20"}),
               _web._wiki, _WURL, 1.0)


# First-word junk from a throttled engine must rank below on-topic hits.
def _junk_get(url, timeout, method="GET", **kw):
    if "api.php" in url:
        return _R(200, data={"query": {"search": [{"title": "Things Fall Apart", "snippet": "novel by Chinua Achebe"}]}})
    if "bing" in url:
        return _R(200, text="".join(f'<li class="b_algo"><h2><a href="https://dict{i}.example/wrote">WROTE definition</a></h2>'
                                    f'<p>past tense of write</p></li>' for i in range(8)))
    return None


_web._get = _junk_get
_e = _web.ENGINES[:]
_web.ENGINES[:] = ["bing", "wiki"]
_hits = _web.search(["Who wrote the novel Things Fall Apart?", "wrote novel things fall apart"], timeout=2.0)
_web.ENGINES[:] = _e
ok = len(_hits) == 9 and "Things_Fall_Apart" in _hits[0].url
fails += not ok
print(f"junk hits demoted: {'ok' if ok else 'FAIL'} first of {len(_hits)} = {_hits[0].url if _hits else None}")
_web._get = _real_get

print(f"\nscore {score}/{len(QS)}   failures {fails}   work dir {work}")
subprocess.run(["pkill", "-f", "mc4/server.py"])
web.terminate()
sys.exit(1 if fails else 0)
