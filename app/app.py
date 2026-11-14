#!/usr/bin/env python3
"""AMD AI League - Match 4 ("Intelligent Web Information Retrieval").

The official I/O contract is not published yet, so this entry point accepts the
shapes the earlier matches used and common variants, all mapped to one pipeline:

    python3 /app/app.py --index                       # optional warm-up: load the model once
    python3 /app/app.py --query-id query_01 --query "question" [--url https://start.page]
    python3 /app/app.py --input /app/input/query_01.json     # {"query": ..., "url": ...}
    python3 /app/app.py --input /app/input/questions.json    # a list, or {"queries": [...]}

Output, one file per question, always written whole (atomic rename), an empty
answer on any failure: /app/output/<query-id>_output.json
    {"answer": "1889", "citations": ["https://..."], "confidence": 0.9}

Nothing is printed to stdout (unless --stdout is given for debugging); logs go to
/tmp/mc4/app.log.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE / "vendor"))   # pip deps from the app layer; base-image packages win

from mc4 import paths  # noqa: E402

SERVER = HERE / "mc4" / "server.py"
STARTUP_DEADLINE_S = 540


# --------------------------------------------------------------------------- model server

def _request(req: dict, timeout: float) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(str(paths.SOCKET))
        s.sendall((json.dumps(req) + "\n").encode("utf-8"))
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.decode("utf-8"))
    finally:
        s.close()


def server_alive() -> bool:
    try:
        return bool(_request({"op": "ping"}, timeout=2).get("ok"))
    except (OSError, ValueError):
        return False


def _starting() -> bool:
    """A server process launched recently and still loading weights."""
    try:
        return time.time() - (paths.WORK_DIR / "server.started").stat().st_mtime < STARTUP_DEADLINE_S
    except OSError:
        return False


def start_server() -> None:
    if server_alive() or _starting():
        return
    paths.WORK_DIR.mkdir(parents=True, exist_ok=True)
    (paths.WORK_DIR / "server.started").write_text(str(time.time()))
    subprocess.Popen(
        [sys.executable, str(SERVER)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=open(paths.WORK_DIR / "server.stderr", "ab"),
        start_new_session=True, close_fds=True, cwd=str(HERE),
    )


class RemoteLLM:
    """Client for the resident server. Fails fast if the model is not ready, and
    the pipeline then falls back to its no-model path."""

    def __init__(self, deadline: float):
        self.deadline = deadline

    def chat(self, system: str, text: str, max_new_tokens: int = 120) -> str:
        while not server_alive():
            if time.time() > self.deadline - 4:
                raise RuntimeError("model server not ready")
            time.sleep(0.5)
        resp = _request({"op": "chat", "system": system, "text": text, "max_new_tokens": max_new_tokens,
                         "deadline": self.deadline - 1.0},
                        timeout=max(2.0, self.deadline - time.time()))
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "server error"))
        return str(resp.get("text", ""))


def warmup() -> None:
    t0 = time.time()
    start_server()
    while time.time() - t0 < STARTUP_DEADLINE_S:
        if server_alive():
            paths.log(f"model ready after {time.time() - t0:.0f}s")
            return
        time.sleep(1.0)
    paths.log("model server did not come up in time")


# --------------------------------------------------------------------------- I/O

def _safe_stem(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("._")[:120]
    return s or "query"


def write_output(out_dir: Path, stem: str, answer: str, citations: list[str], confidence: float,
                 to_stdout: bool = False) -> None:
    rec = {"answer": answer, "citations": citations if answer else [], "confidence": round(float(confidence), 3)}
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{_safe_stem(stem)}_output.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out)
    if to_stdout:
        print(json.dumps(rec, ensure_ascii=False), flush=True)


def _item(d: dict, i: int, default_stem: str) -> tuple[str, str, str | None]:
    q = next((d[k] for k in ("query", "question", "prompt", "task", "request", "q", "input") if isinstance(d.get(k), str)), "")
    u = next((d[k] for k in ("url", "start_url", "startUrl", "website", "site", "link") if isinstance(d.get(k), str)), None)
    sid = next((d[k] for k in ("query_id", "id", "question_id", "qid", "task_id", "name") if d.get(k) not in (None, "")), None)
    return (str(sid) if sid is not None else default_stem if i < 0 else f"query_{i + 1:02d}"), q, u


def load_input(path: Path) -> list[tuple[str, str, str | None]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(text)
    except ValueError:
        if path.suffix.lower() in (".txt", ".md", ""):
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if len(lines) == 1:
                return [(path.stem, lines[0], None)]
            return [(f"query_{i + 1:02d}", ln, None) for i, ln in enumerate(lines)]
        raise
    if isinstance(data, dict):
        for k in ("queries", "questions", "tasks", "items", "data"):
            if isinstance(data.get(k), list):
                data = data[k]
                break
    if isinstance(data, dict):
        return [_item(data, -1, path.stem)]
    if isinstance(data, list):
        out = []
        for i, d in enumerate(data):
            if isinstance(d, str):
                out.append((f"query_{i + 1:02d}", d, None))
            elif isinstance(d, dict):
                out.append(_item(d, i, path.stem))
        return out
    if isinstance(data, str):
        return [(path.stem, data, None)]
    return []


def solve(stem: str, question: str, url: str | None, out_dir: Path, to_stdout: bool, no_llm: bool) -> None:
    t0 = time.time()
    try:
        from mc4.pipeline import answer
        llm = None if no_llm else RemoteLLM(t0 + paths.DEADLINE_S)
        a, cites, conf = answer(question, url, paths.DEADLINE_S, llm)
    except Exception as e:  # noqa: BLE001 - always write a valid file
        paths.log(f"{stem}: failed {type(e).__name__}: {e}")
        a, cites, conf = "", [], 0.0
    write_output(out_dir, stem, a, cites, conf, to_stdout)
    paths.log(f"{stem}: {time.time() - t0:.1f}s answer={a!r} cites={cites}")


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--index", "--warmup", "--setup", "--init", dest="warm", nargs="?", const=True, default=None)
    ap.add_argument("--query", "--question", "--prompt", "--task", "-q", dest="query")
    ap.add_argument("--query-id", "--query_id", "--id", "--question-id", "--qid", "--task-id", dest="qid")
    ap.add_argument("--url", "--start-url", "--start_url", "--website", "--site", dest="url")
    ap.add_argument("--input", "--input-file", "--input-json", "--tasks-file", "--questions", "-i", dest="input")
    ap.add_argument("--output-dir", "--output", "--out", "-o", dest="out")
    ap.add_argument("--corpus", help="accepted and ignored (Match 3 shape)")
    ap.add_argument("--stdout", action="store_true", help="debug: also print each result")
    ap.add_argument("--no-llm", action="store_true", help="debug: retrieval only, no model")
    args, rest = ap.parse_known_args()

    out_dir = Path(args.out) if args.out and not args.out.endswith(".json") else paths.OUTPUT_DIR
    if args.out and args.out.endswith(".json"):
        out_dir = Path(args.out).parent

    if not args.no_llm:
        start_server()                     # idempotent; model loads in the background
    if args.warm is not None and not args.query and not args.input:
        if not args.no_llm:
            warmup()
        return 0

    jobs: list[tuple[str, str, str | None]] = []
    if args.input:
        try:
            jobs = load_input(Path(args.input))
        except Exception as e:  # noqa: BLE001
            paths.log(f"bad input {args.input}: {e}")
            jobs = [(Path(args.input).stem, "", None)]
    query = args.query or " ".join(r for r in rest if not r.startswith("-")).strip()
    if query:
        jobs.append((args.qid or "query", query, args.url))
    elif args.qid and not jobs:
        jobs.append((args.qid, "", args.url))

    for stem, q, u in jobs:
        if args.qid and len(jobs) == 1:
            stem = args.qid
        solve(stem, q, u or args.url, out_dir, args.stdout, args.no_llm)
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    finally:
        # Hard exit: stray fetch threads (already past their deadline) must not keep
        # the process alive after the output files are written.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc if isinstance(rc, int) else 0)
