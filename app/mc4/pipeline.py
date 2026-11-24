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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from . import paths, web
from .rank import Chunk, chunk_page, rank, rank_links, terms

URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
SKIP_HOSTS = ("youtube.com", "youtu.be", "facebook.com", "instagram.com", "tiktok.com", "x.com",
              "twitter.com", "pinterest.", "linkedin.com")
CONTEXT_WORDS = 1500
EXPAND_CAP_S = 3.0          # the query-rewrite call never holds up searching for longer than this

SYSTEM = ("You answer questions using only the numbered web sources you are given. "
          "Reply with one JSON object and nothing else.")

_HEAD = """QUESTION: {q}

SOURCES:
{sources}

Rules:
- "answer": the shortest exact answer: a name, number, date, version or short phrase, copied from the source wording. Keep a unit only when the question needs it. Never a full sentence, never a label like "Answer:".
- Answer exactly what is asked: for "who" or "which" give the name itself, not a description of it.
- If the question asks for several items, or asks two things (for example who and when), give all of them separated by ", ".
- If sources disagree, prefer the official or primary source and the most recent figure. Ignore entries described as former, previous or brief.
- "source": the id of the source that states the answer, e.g. "S2".
- "quote": the exact words (at most 25) from that source that contain the answer.
"""
_TAIL = """
JSON: {{"found": true, "answer": "...", "source": "S1", "quote": "..."}}"""

# First read: the model may ask for one more lookup when the question needs two steps.
PROMPT = _HEAD + """- If the question is about something that must first be identified (for example "the year the university where X studied was founded") and the sources identify it but do not state the final fact about it, reply {{"found": false, "search": "a short web search query for the missing fact"}}.
- If no source contains the answer, reply {{"found": false}}.
""" + _TAIL
# Last read: answer or say not found.
PROMPT_FINAL = _HEAD + """- If no source contains the answer, reply {{"found": false}}.
""" + _TAIL

EXPAND = """Write one short web search query (at most 8 words) that would find the answer to this question. If the question refers to something indirectly (for example "the university where X studied" or "the element named after Y's homeland"), name that thing directly using what you know. Reply with the query only.

QUESTION: {q}"""

REFORMULATE = """Write one short web search query (at most 10 words) that would find the answer to this question. Reply with the query only.

QUESTION: {q}"""


def norm(s: str) -> str:
    s = s.lower().replace("–", "-").replace("—", "-")
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)          # 1,234 -> 1234
    return re.sub(r"[^0-9a-zÀ-￿]+", "", s)


# -- grounding ------------------------------------------------------------------
# An answer is grounded when it appears in the source text. Numbers are compared as
# whole numbers ("12" is not found inside "2012" or "1,200"), and a few exact
# equivalences are accepted: the same calendar date written another way,
# percent / %, and British / American unit spellings. Nothing here is fuzzy.

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})
_MONTHS["sept"] = 9
_MON = "(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_DAY = r"(\d{1,2})(?:st|nd|rd|th)?"
_DATE_MDY = re.compile(rf"\b{_MON}\s+{_DAY}\s*,?\s+(\d{{4}})\b")
_DATE_DMY = re.compile(rf"\b{_DAY}\s+(?:of\s+)?{_MON}\s*,?\s+(\d{{4}})\b")
_DATE_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_TOKEN = re.compile(r"\d+(?:\.\d+)*|[a-zÀ-\uffff]+")
_SPELLING = {
    "metre": "meter", "metres": "meters", "kilometre": "kilometer", "kilometres": "kilometers",
    "centimetre": "centimeter", "centimetres": "centimeters", "millimetre": "millimeter",
    "millimetres": "millimeters", "litre": "liter", "litres": "liters",
    "millilitre": "milliliter", "millilitres": "milliliters",
    "tonne": "ton", "tonnes": "tons", "pct": "percent",
}


def _iso(y: str, m: int, d: str) -> str:
    return f" isodate {int(y):04d}{m:02d}{int(d):02d} " if 1 <= m <= 12 and 1 <= int(d) <= 31 else ""


def _dates(s: str) -> str:
    """Rewrite every recognisable calendar date as one canonical token pair."""
    s = _DATE_MDY.sub(lambda m: _iso(m.group(3), _MONTHS[m.group(1)], m.group(2)) or m.group(0), s)
    s = _DATE_DMY.sub(lambda m: _iso(m.group(3), _MONTHS[m.group(2)], m.group(1)) or m.group(0), s)
    return _DATE_ISO.sub(lambda m: _iso(m.group(1), int(m.group(2)), m.group(3)) or m.group(0), s)


def _tokens(s: str, dates: bool = False) -> list[str]:
    s = s.lower().replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)          # 1,234 -> 1234
    if dates:
        s = _dates(s)
    s = s.replace("%", " percent ").replace("per cent", "percent")
    out = []
    for t in _TOKEN.findall(s):
        if t[0].isdigit() and t.count(".") == 1:      # 12.50 -> 12.5, 3.0 -> 3
            t = t.rstrip("0").rstrip(".")
        out.append(_SPELLING.get(t, t))
    return out


def _rep(s: str, dates: bool = False) -> str:
    """Comparable form: words run together (so "Wi-Fi" equals "WiFi"), every number
    fenced as |n| (so a number only ever matches a whole number)."""
    return "".join(f"|{t}|" if t[0].isdigit() else t for t in _tokens(s, dates))


def _one_grounded(answer: str, text: str) -> bool:
    a = _rep(answer)
    if not a:
        return False
    if a in _rep(text):
        return True
    a_d = _rep(answer, dates=True)                    # same date, written differently
    return "isodate" in a_d and a_d in _rep(text, dates=True)


def grounded(answer: str, text: str) -> bool:
    if not norm(answer):
        return False
    if _one_grounded(answer, text):
        return True
    parts = [p for p in re.split(r",|;| and ", answer) if norm(p)]
    return len(parts) > 1 and all(_one_grounded(p, text) for p in parts)


def _mostly(a: str, text: str, share: float) -> bool:
    """At least `share` of a's tokens are tokens of text (order ignored)."""
    ta = _tokens(a)
    if not ta:
        return False
    tt = set(_tokens(text))
    return sum(1 for t in ta if t in tt) >= share * len(ta)


def partly_grounded(answer: str, text: str) -> bool:
    """A list answer of three or more items where at least two thirds of the items
    appear in the text (a source often spells one name of a long list differently)."""
    parts = [p for p in re.split(r",|;| and ", answer) if norm(p)]
    return len(parts) >= 3 and sum(1 for p in parts if _one_grounded(p, text)) * 3 >= len(parts) * 2


def clean_answer(a: str) -> str:
    a = str(a or "").strip().strip('"').strip()
    a = re.sub(r"^(the )?answer( is)?\s*[:\-]?\s*", "", a, flags=re.I)
    if a.endswith(".") and not re.search(r"\b[A-Z][a-z]{0,3}\.$", a):   # keep "Inc." "Jr."
        a = a[:-1]
    return a.strip()


_FIELD = r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"'


NON_ANSWER_RE = re.compile(r"(?:n/?a|none|null|unknown|not (?:stated|mentioned|specified|given|provided|available|found|"
                           r"known|listed|clear)(?: (?:in|on) the (?:sources?|pages?|text))?|no (?:answer|information|data)"
                           r"(?: (?:available|found|given))?|unclear|cannot be determined)[.!]?", re.I)


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            v = json.loads(m.group(0))
            return v if isinstance(v, dict) else {}
        except ValueError:
            pass
    # A long list answer can use up the output budget before the closing brace:
    # keep the fields that were written completely.
    if re.search(r'"found"\s*:\s*false', text or ""):
        f = re.search(_FIELD % "search", text or "")
        return {"found": False, "search": f.group(1)} if f else {"found": False}
    out = {}
    for k in ("answer", "source", "quote"):
        f = re.search(_FIELD % k, text or "")
        if f:
            try:
                out[k] = json.loads(f'"{f.group(1)}"')
            except ValueError:
                out[k] = f.group(1)
    return out if out.get("answer") else {}


def _start_urls(question: str, url: str | None) -> list[str]:
    urls = ([url] if url else []) + URL_RE.findall(question)
    return list(dict.fromkeys(u.rstrip(".,;") for u in urls if u))


# Words that point at the given page ("According to this page, ...") mean nothing to a
# search engine and push the real terms down: drop them from search queries only.
DEICTIC_RE = re.compile(r"\b(according to|as stated (on|in)|based on|on|in|from|per)\s+(this|the given|the linked|the above)\s+"
                        r"(web ?page|page|site|website|article|document|doc|link|url|post|source)\b[\s,:]*", re.I)


def _queries(question: str) -> list[str]:
    q = DEICTIC_RE.sub(" ", URL_RE.sub("", question))
    q = " ".join(q.split()).strip(" ,")
    q = q[:1].upper() + q[1:]
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
        self.next_query = ""                          # follow-up search the model asked for
        self.extra = ""                               # the model's own search query for the question
        self.unverified: tuple[str, list[str], float] | None = None   # answer not found on any page read

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
        if self.extra:
            qs.append(self.extra)
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

    def expand(self) -> None:
        """One short model call: a direct search query, with indirect references
        resolved ("the university where Turing earned his PhD" -> "Princeton")."""
        ready = getattr(self.llm, "ready", None)
        if ready is not None and not ready():         # still loading: searching cannot wait for it
            paths.log("expand skipped: model not ready")
            return
        box: dict = {}

        def run() -> None:
            try:
                box["raw"] = self.llm.chat("", EXPAND.format(q=self.q), max_new_tokens=24)
            except Exception as e:  # noqa: BLE001
                box["err"] = e
        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(min(EXPAND_CAP_S, max(0.0, self.left() - 15)))
        if "raw" not in box:                          # slow or failed: search with the question alone
            paths.log(f"expand gave up: {box.get('err') or 'over the time cap'}")
            return
        raw = box["raw"]
        line = next((ln for ln in str(raw).splitlines() if ln.strip()), "")
        x = " ".join(URL_RE.sub("", line).strip().strip('"\'').split()[:10])[:120]
        if norm(x) and norm(x) not in norm(self.q) and norm(self.q) not in norm(x):
            self.extra = x
            paths.log(f"extra query: {x!r}")

    def gather(self) -> None:
        if self.llm is not None and not self.starts:  # a given page is the place to look: no rewriting
            self.expand()
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
    def context(self, extra: str = "") -> list[Chunk]:
        picked, words, texts = [], 0, set()
        for c in rank(f"{self.q} {self.extra} {extra}".strip(), list(self.chunks)):
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

    def ask(self, ctx: list[Chunk], final: bool = False) -> tuple[str, list[str], float]:
        if not ctx:
            return "", [], 0.0
        src = "\n\n".join(f"[S{i + 1}] {c.title[:90]} ({c.url})\n{c.text}" for i, c in enumerate(ctx))
        raw = self.llm.chat(SYSTEM, (PROMPT_FINAL if final else PROMPT).format(q=self.q, sources=src),
                            max_new_tokens=160)
        d = parse_json(raw)
        paths.log(f"llm: {raw[:300]!r}")
        if not d or d.get("found") is False:
            nq = d.get("search") if d else ""
            self.next_query = " ".join(str(nq).split())[:200] if isinstance(nq, str) else ""
            return "", [], 0.0
        ans = clean_answer(d.get("answer", ""))
        if not ans or re.fullmatch(r"\[?S\d+\]?", ans):        # empty, or a source id instead of an answer
            return "", [], 0.0
        m = re.search(r"\d+", str(d.get("source", "")))
        cited = ctx[int(m.group(0)) - 1] if m and 0 < int(m.group(0)) <= len(ctx) else None
        if not norm(ans):                             # punctuation-only answer, e.g. a separator: literal match
            lit = " ".join(ans.split())
            for c in ([cited] if cited else []) + ctx:
                if lit and lit in " ".join(c.text.split()):
                    return ans, [c.url], 0.6
            paths.log(f"ungrounded symbol answer dropped: {ans!r}")
            return "", [], 0.0
        if cited and grounded(ans, cited.text):
            return ans, [self._site_cite(ans, cited.url)], 0.9
        quote = str(d.get("quote", ""))
        for c in ctx + self.chunks:                   # grounded elsewhere: cite that page instead
            if grounded(ans, c.text):
                return ans, [self._site_cite(ans, c.url)], 0.7
        if cited and quote and norm(quote) and norm(quote) in norm(cited.text) and not norm(ans) in norm(self.q):
            return ans, [cited.url], 0.5             # paraphrased value, quote is real
        if cited and partly_grounded(ans, cited.text):
            return ans, [cited.url], 0.5             # long list, most items on the cited page
        if (cited and len(_tokens(quote)) >= 5 and _mostly(quote, cited.text, 0.9)
                and _mostly(ans, quote + " " + cited.text, 0.8) and not norm(ans) in norm(self.q)):
            return ans, [cited.url], 0.4             # quote slightly reworded, answer words are on the page
        paths.log(f"ungrounded answer set aside: {ans!r}")
        if self.unverified is None and not norm(ans) in norm(self.q) and not NON_ANSWER_RE.fullmatch(ans):
            self.unverified = (ans, [cited.url] if cited else [ctx[0].url], 0.2)
        return "", [], 0.0                            # not on any page we read

    def _site_cite(self, ans: str, url: str) -> str:
        """A start URL was given: cite a page of that site when the answer is on one."""
        if not self.starts:
            return url
        host = urlparse(self.starts[0]).netloc.lower().removeprefix("www.")
        if urlparse(url).netloc.lower().removeprefix("www.").endswith(host):
            return url
        for c in self.chunks:
            if urlparse(c.url).netloc.lower().removeprefix("www.").endswith(host) and grounded(ans, c.text):
                paths.log(f"citation moved to the given site: {c.url}")
                return c.url
        return url

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
    if a:
        return a, cites, conf
    # Last resort when nothing better turns up: the model's answer that could not be
    # found on a page, marked with low confidence (an empty answer can never be right).
    last = s.unverified or ("", [], 0.0)
    if s.left() < 9:
        return last
    # Second chance: follow links from the best pages, or search again with a better query.
    rq = s.next_query
    if s.pages and not rq:                            # the model named the missing fact: search, do not crawl
        s.hop()
    if s.left() > 12:
        rq = rq or s.reformulate()
        if rq:
            hits = web.search([rq], timeout=min(4.0, s.left() - 6))
            for i, h in enumerate(hits[:8]):
                if h.snippet:
                    s.chunks.append(Chunk(h.url, h.title, h.snippet, 45 + i))
            urls = [h.url for h in hits if h.url not in s.seen][:5]
            s.seen.update(urls)
            s._add(web.fetch_many(urls, min(time.time() + 6, s.t_end - 5)), 40)
    if s.left() < 3:
        return last
    try:
        a, cites, conf = s.ask(s.context(rq), final=True)
    except Exception as e:  # noqa: BLE001
        paths.log(f"llm failed (hop): {e}")
        return last
    return (a, cites, conf) if a else (s.unverified or ("", [], 0.0))
