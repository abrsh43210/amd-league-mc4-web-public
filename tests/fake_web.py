#!/usr/bin/env python3
"""Offline stand-in for the web, for tests: serves tests/web/ as static pages and
fakes two search engines (DuckDuckGo-HTML and Bing result markup, including
DDG's redirect links) that rank the indexed pages by word overlap.

    python3 tests/fake_web.py PORT
"""
import html
import json
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

ROOT = Path(__file__).resolve().parent / "web"
# Pages a search engine "knows". The shop is deliberately NOT indexed: reaching it needs crawling.
INDEXED = {
    "eiffel.html": None, "old_news.html": None, "city.html": None,
    "orrery_report_2025.pdf": "Orrery Systems annual report 2025 total revenue fiscal 2025",
}


def page_text(name):
    if INDEXED[name]:
        return INDEXED[name]
    return re.sub(r"<[^>]+>", " ", (ROOT / name).read_text())


def words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower())) - {"the", "a", "of", "in", "is", "what", "how", "was", "did"}


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT), **k)

    def log_message(self, *a):
        pass

    def _results(self):
        q = parse_qs(urlparse(self.path).query).get("q", [""])[0]
        base = f"http://{self.headers['Host']}"
        scored = []
        for name in INDEXED:
            s = len(words(q) & words(page_text(name)))
            if s:
                scored.append((s, name))
        scored.sort(reverse=True)
        return base, [(f"{base}/{n}", n, " ".join(page_text(n).split())[:160]) for _, n in scored]

    def _send(self, body, ctype="text/html"):
        b = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/ddg":
            _, res = self._results()
            items = "".join(
                f'<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg={quote(u, safe="")}&rut=x">{html.escape(t)}</a>'
                f'<a class="result__snippet">{html.escape(s)}</a></div>' for u, t, s in res)
            items = '<div class="result result--ad"><a class="result__a" href="https://ads.example/x">AD</a></div>' + items
            return self._send(f"<html><body>{items}</body></html>")
        if p == "/bing":
            _, res = self._results()
            items = "".join(f'<li class="b_algo"><h2><a href="{u}">{html.escape(t)}</a></h2>'
                            f'<div class="b_caption"><p>{html.escape(s)}</p></div></li>' for u, t, s in res[::-1])
            return self._send(f"<html><body><ol>{items}</ol></body></html>")
        if p == "/w/api.php":
            return self._send(json.dumps({"query": {"search": []}}), "application/json")
        return super().do_GET()


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H)
    srv.serve_forever()
