"""Chunking and BM25 ranking of page text against the question."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

STOP = set("""a an the of in on at to for from by with and or is are was were be been being do does did
what which who whom whose when where why how many much is it its this that these those as about into
than then there their they he she his her you your i we our me my can could would should will shall
please tell find give list name according page website site show me state""".split())

_TOK = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*", re.I)


def tokens(text: str) -> list[str]:
    return [t for t in (m.group(0).lower() for m in _TOK.finditer(text)) if t not in STOP]


def terms(text: str) -> list[str]:
    return list(dict.fromkeys(tokens(text)))


@dataclass
class Chunk:
    url: str
    title: str
    text: str
    order: int          # source order (search rank / crawl order): lower is better
    score: float = 0.0


def chunk_page(url: str, title: str, lines: list[str], order: int, max_words: int = 90) -> list[Chunk]:
    chunks, buf, n = [], [], 0
    for ln in lines:
        w = len(ln.split())
        if w > max_words * 2:                      # a wall of text: split it on sentences
            for s in re.split(r"(?<=[.!?])\s+", ln):
                sw = len(s.split())
                if n + sw > max_words and buf:
                    chunks.append(" ".join(buf))
                    buf, n = [], 0
                buf.append(s)
                n += sw
            continue
        if n + w > max_words and buf:
            chunks.append(" ".join(buf))
            buf, n = [], 0
        buf.append(ln)
        n += w
    if buf:
        chunks.append(" ".join(buf))
    return [Chunk(url, title, c, order) for c in chunks if len(c) > 15]


_NUMQ = re.compile(r"\b(how many|how much|how long|how tall|how old|what year|which year|when|what date|population|number of|price|cost|height|length|percent)\b", re.I)


def rank(question: str, chunks: list[Chunk]) -> list[Chunk]:
    q = terms(question)
    if not chunks:
        return []
    docs = [tokens(c.title + " " + c.text) for c in chunks]
    N = len(docs)
    avg = sum(len(d) for d in docs) / N or 1.0
    df = {t: sum(1 for d in docs if t in d) for t in q}
    wants_num = bool(_NUMQ.search(question))
    for c, d in zip(chunks, docs):
        tf: dict[str, int] = {}
        for t in d:
            if t in df:
                tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t in q:
            f = tf.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len(d) / avg))
        cover = sum(1 for t in q if t in tf) / max(1, len(q))
        s *= 0.6 + cover                             # reward chunks that cover more of the question
        if wants_num and re.search(r"\d", c.text):
            s *= 1.15
        s *= 1.0 / (1.0 + 0.04 * c.order)            # mild prior for higher search ranks
        c.score = s
    return sorted(chunks, key=lambda c: -c.score)


def rank_links(question: str, links: list[tuple[str, str]], same_host: str | None, seen: set[str]) -> list[str]:
    """Order a page's links by how well their anchor text + path match the question."""
    from urllib.parse import urlparse
    q = set(terms(question))
    scored = []
    for u, anchor in links:
        if u in seen:
            continue
        p = urlparse(u)
        if same_host and p.netloc.lower().removeprefix("www.") != same_host:
            continue
        toks = set(tokens(anchor + " " + p.path.replace("/", " ").replace("-", " ").replace("_", " ")))
        s = len(q & toks)
        scored.append((s, -len(u), u))                # unmatched same-site links stay as a last resort
    scored.sort(reverse=True)
    if not same_host:
        scored = [x for x in scored if x[0] > 0]
    return list(dict.fromkeys(u for _, _, u in scored))
