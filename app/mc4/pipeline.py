"""Question -> (answer, citations, confidence).

1. Seeds: a start URL (from --url or found in the question) is crawled best-first
   on its own site; in parallel, several search engines are queried and fused.
2. Pages are fetched in parallel, turned into text (tables become "label: value"
   lines) and cut into ~90-word chunks.
3. BM25 picks the best chunks; the local LLM on the AMD GPU reads only those and
   returns strict JSON: answer, source id, supporting quote.
4. Grounding: the answer must appear in the cited chunk (or another fetched chunk,
   which then becomes the citation). Not found -> one more hop through the most
   relevant links, then an honest empty answer.
Every step respects one deadline so the per-question budget is never blown.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from . import paths, web
from .rank import Chunk, chunk_page, rank, rank_links, terms

URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
SKIP_HOSTS = ("youtube.com", "youtu.be", "facebook.com", "instagram.com", "tiktok.com", "x.com",
              "twitter.com", "pinterest.", "linkedin.com")
CONTEXT_WORDS = 1500

SYSTEM = ("You answer questions using only the numbered web sources you are given. "
          "Reply with one JSON object and nothing else.")

PROMPT = """QUESTION: {q}

SOURCES:
{sources}

Rules:
- "answer": the shortest exact answer: a name, number, date, version or short phrase, copied from the source wording. Keep a unit only when the question needs it. Never a full sentence, never a label like "Answer:".
- If the question asks for several items, separate them with ", ".
- If sources disagree, prefer the official or primary source and the most recent figure.
- "source": the id of the source that states the answer, e.g. "S2".
- "quote": the exact words (at most 25) from that source that contain the answer.
- If no source contains the answer, reply {{"found": false}}.

JSON: {{"found": true, "answer": "...", "source": "S1", "quote": "..."}}"""

REFORMULATE = """Write one short web search query (at most 10 words) that would find the answer to this question. Reply with the query only.

QUESTION: {q}"""


def norm(s: str) -> str:
    s = s.lower().replace("–", "-").replace("—", "-")
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)          # 1,234 -> 1234
    return re.sub(r"[^0-9a-zÀ-￿]+", "", s)


def grounded(answer: str, text: str) -> bool:
    a = norm(answer)
    if not a:
        return False
    t = norm(text)
    if a in t:
        return True
    parts = [norm(p) for p in re.split(r",|;| and ", answer) if norm(p)]
    return len(parts) > 1 and all(p in t for p in parts)


def clean_answer(a: str) -> str:
    a = str(a or "").strip().strip('"').strip()
    a = re.sub(r"^(the )?answer( is)?\s*[:\-]?\s*", "", a, flags=re.I)
    if a.endswith(".") and not re.search(r"\b[A-Z][a-z]{0,3}\.$", a):   # keep "Inc." "Jr."
        a = a[:-1]
    return a.strip()


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {}
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, dict) else {}
    except ValueError:
        return {}


def _start_urls(question: str, url: str | None) -> list[str]:
    urls = ([url] if url else []) + URL_RE.findall(question)
    return list(dict.fromkeys(u.rstrip(".,;") for u in urls if u))


def _queries(question: str) -> list[str]:
    q = URL_RE.sub("", question).strip()
    kw = " ".join(terms(q)[:12])
    out = [q[:300]]
    if kw and kw.lower() != q.lower():
        out.append(kw)
    return out


class Session:
    """State for one question: pages seen, chunks gathered."""

    def __init__(self, question: str, start_url: str | None, deadline_s: float, llm=None):
        self.q = question.strip()
        self.t_end = time.time() + deadline_s
        self.llm = llm
        self.starts = _start_urls(self.q, start_url)
        self.pages: dict[str, web.Page] = {}
        self.chunks: list[Chunk] = []
        self.seen: set[str] = set()
        self.tokens_used = 0
        self.hits: list = []

    def left(self) -> float:
        return self.t_end - time.time()

    def _add(self, pages: list[web.Page], base_order: int) -> None:
        for i, p in enumerate(pages):
            if p.url in self.pages:
                continue
            self.pages[p.url] = p
            self.chunks += chunk_page(p.url, p.title, p.lines, base_order + i)

    # -- gathering --------------------------------------------------------------
    def _crawl_start(self) -> list[web.Page]:
        out = []
        for u in self.starts[:2]:
            self.seen.add(u)
            p = web.fetch(u, min(7.0, max(1.0, self.left() - 10)))
            if not p.ok:
                continue
            out.append(p)
            host = urlparse(p.url).netloc.lower().removeprefix("www.")
            nxt = rank_links(self.q, p.links, host, self.seen)[:6]
            self.seen.update(nxt)
            out += web.fetch_many(nxt, min(time.time() + 7, self.t_end - 9))
        return out

    def _search(self) -> list[web.Page]:
        qs = _queries(self.q)
        if self.starts:
            host = urlparse(self.starts[0]).netloc.removeprefix("www.")
            qs.append(f"site:{host} " + " ".join(terms(self.q)[:8]))
        hits = web.search(qs, timeout=min(5.0, max(1.0, self.left() - 14)))
        self.hits = hits
        paths.log(f"search: {len(hits)} fused hits")
        for i, h in enumerate(hits[:12]):
            if h.snippet:
                self.chunks.append(Chunk(h.url, h.title, h.snippet, 50 + i))
        urls = [h.url for h in hits if not any(s in h.url for s in SKIP_HOSTS) and h.url not in self.seen]
        urls = urls[: paths.MAX_PAGES]
        self.seen.update(urls)
        return web.fetch_many(urls, min(time.time() + 8, self.t_end - 8))

    def gather(self) -> None:
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_start = ex.submit(self._crawl_start) if self.starts else None
            f_search = ex.submit(self._search)
            start_pages = f_start.result() if f_start else []
            search_pages = f_search.result()
        self._add(start_pages, 0)
        self._add(search_pages, len(start_pages))
        paths.log(f"gathered {len(self.pages)} pages, {len(self.chunks)} chunks, {self.left():.1f}s left")

    def hop(self) -> None:
        """Follow the most relevant links of the best pages (one level)."""
        best = []
        for c in rank(self.q, list(self.chunks))[:6]:
            if c.url in self.pages and c.url not in best:
                best.append(c.url)
        cand = []
        for u in best[:3]:
            p = self.pages[u]
            cand += rank_links(self.q, p.links, urlparse(p.url).netloc.lower().removeprefix("www."), self.seen)[:3]
        cand = list(dict.fromkeys(cand))[:5]
        self.seen.update(cand)
        self._add(web.fetch_many(cand, min(time.time() + 6, self.t_end - 6)), 30)

    # -- reading -------------------------------------------------------------------
    def context(self) -> list[Chunk]:
        picked, words, texts = [], 0, set()
        for c in rank(self.q, list(self.chunks)):
            key = norm(c.text)[:200]
            if key in texts or c.score <= 0:
                continue
            w = len(c.text.split())
            if words + w > CONTEXT_WORDS:
                continue
            picked.append(c)
            texts.add(key)
            words += w
            if len(picked) >= 16:
                break
        return picked

    def ask(self, ctx: list[Chunk]) -> tuple[str, list[str], float]:
        if not ctx:
            return "", [], 0.0
        src = "\n\n".join(f"[S{i + 1}] {c.title[:90]} ({c.url})\n{c.text}" for i, c in enumerate(ctx))
        raw = self.llm.chat(SYSTEM, PROMPT.format(q=self.q, sources=src), max_new_tokens=120)
        d = parse_json(raw)
        paths.log(f"llm: {raw[:300]!r}")
        if not d or d.get("found") is False:
            return "", [], 0.0
        ans = clean_answer(d.get("answer", ""))
        if not ans or re.fullmatch(r"\[?S\d+\]?", ans):        # empty, or a source id instead of an answer
            return "", [], 0.0
        m = re.search(r"\d+", str(d.get("source", "")))
        cited = ctx[int(m.group(0)) - 1] if m and 0 < int(m.group(0)) <= len(ctx) else None
        if cited and grounded(ans, cited.text):
            return ans, [cited.url], 0.9
        quote = str(d.get("quote", ""))
        for c in ctx + self.chunks:                   # grounded elsewhere: cite that page instead
            if grounded(ans, c.text):
                return ans, [c.url], 0.7
        if cited and quote and norm(quote) and norm(quote) in norm(cited.text) and not norm(ans) in norm(self.q):
            return ans, [cited.url], 0.5             # paraphrased value, quote is real
        paths.log(f"ungrounded answer dropped: {ans!r}")
        return "", [], 0.0                            # not on any page we read: do not guess

    def heuristic(self, ctx: list[Chunk]) -> tuple[str, list[str], float]:
        """No model available: return the best-matching sentence from the best chunk."""
        if not ctx:
            return "", [], 0.0
        q = set(terms(self.q))
        best, bs = "", -1
        for s in re.split(r"(?<=[.!?;])\s+", ctx[0].text):
            sc = len(q & set(terms(s)))
            if sc > bs:
                best, bs = s, sc
        return " ".join(best.split()[:40]), [ctx[0].url], 0.1

    def reformulate(self) -> str:
        try:
            return self.llm.chat("", REFORMULATE.format(q=self.q), max_new_tokens=24).strip().strip('"')[:200]
        except Exception:  # noqa: BLE001
            return ""


def answer(question: str, start_url: str | None = None, deadline_s: float | None = None,
           llm=None) -> tuple[str, list[str], float]:
    s = Session(question, start_url, deadline_s or paths.DEADLINE_S, llm)
    if not s.q:
        return "", [], 0.0
    s.gather()
    ctx = s.context()
    if llm is None:
        return s.heuristic(ctx)
    try:
        a, cites, conf = s.ask(ctx) if ctx else ("", [], 0.0)
    except Exception as e:  # noqa: BLE001
        paths.log(f"llm failed: {type(e).__name__}: {e}")
        return s.heuristic(ctx)
    if a or s.left() < 9:
        return a, cites, conf
    # Second chance: follow links from the best pages, or search again with a better query.
    if s.pages:
        s.hop()
    if s.left() > 12:
        rq = s.reformulate()
        if rq:
            hits = web.search([rq], timeout=min(4.0, s.left() - 6))
            urls = [h.url for h in hits if h.url not in s.seen][:5]
            s.seen.update(urls)
            s._add(web.fetch_many(urls, min(time.time() + 6, s.t_end - 5)), 40)
    if s.left() < 3:
        return "", [], 0.0
    try:
        return s.ask(s.context())
    except Exception as e:  # noqa: BLE001
        paths.log(f"llm failed (hop): {e}")
        return "", [], 0.0
