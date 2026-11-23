"""Web access: search engines, page fetching and HTML/PDF to text.

Everything here is defensive: every network call has a timeout, every parser
tolerates garbage, and nothing raises past the public functions. The grading
box has outbound network (Mini Challenge rules, Section 2), but any one search
engine may block datacenter IPs, so several are queried in parallel and fused.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from urllib.parse import parse_qs, quote, quote_plus, unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from . import paths

UA = os.environ.get(
    "MC4_UA",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36",
)
WIKI_UA = os.environ.get("MC4_WIKI_UA", "MC4WebRetrievalAgent/1.0 (AMD AI League entry; python-requests)")
HEADERS = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9",
           "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8"}
MAX_BYTES = 4_000_000

# Overridable so the offline tests can point every engine at a local fixture server.
DDG_URL = os.environ.get("MC4_DDG_URL", "https://html.duckduckgo.com/html/?q={q}")
BING_URL = os.environ.get("MC4_BING_URL", "https://www.bing.com/search?q={q}&setlang=en&cc=us")
MOJEEK_URL = os.environ.get("MC4_MOJEEK_URL", "https://www.mojeek.com/search?q={q}")
WIKI_API = os.environ.get("MC4_WIKI_API", "https://en.wikipedia.org/w/api.php")
ENGINES = [e for e in os.environ.get("MC4_ENGINES", "ddg,bing,wiki,yahoo,mojeek").split(",") if e]

_session = requests.Session()
_session.headers.update(HEADERS)
_adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=0)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)


@dataclass
class Hit:
    url: str
    title: str = ""
    snippet: str = ""
    engine: str = ""
    rank: int = 0


@dataclass
class Page:
    url: str
    title: str = ""
    lines: list[str] = field(default_factory=list)
    links: list[tuple[str, str]] = field(default_factory=list)  # (absolute url, anchor text)
    ok: bool = False


# --------------------------------------------------------------------------- search

def _get_blocking(url: str, timeout: float, end: float, box: dict, method: str = "GET", **kw) -> None:
    if "wikipedia.org" in url or "wikimedia.org" in url:
        kw.setdefault("headers", {"User-Agent": WIKI_UA})   # Wikimedia blocks anonymous browser-like UAs
    r = _session.request(method, url, timeout=(min(3.0, timeout), max(0.5, timeout)), stream=True, **kw)
    box["r"] = r
    r.raw.decode_content = True
    read = getattr(r.raw, "read1", None) or (lambda n: r.raw.read(4096))
    body = bytearray()
    while len(body) <= MAX_BYTES and time.time() < end:
        chunk = read(65536)
        if not chunk:
            break
        body += chunk
    r._content = bytes(body[:MAX_BYTES])
    r._content_consumed = True
    box["ok"] = True


def _get(url: str, timeout: float, method: str = "GET", **kw) -> requests.Response | None:
    """GET with a hard wall-clock cap. requests' timeout is per socket read, so a
    server trickling bytes could run far past the budget: the download runs in a
    daemon thread and is abandoned (and its socket closed) when time is up. The
    entry point exits with os._exit, so abandoned threads never delay the process."""
    import threading
    end = time.time() + max(0.5, timeout)
    box: dict = {}

    def run():
        try:
            _get_blocking(url, timeout, end, box, method, **kw)
        except Exception as e:  # noqa: BLE001 - network: anything can happen
            box["err"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(max(0.1, end - time.time()) + 0.2)
    r = box.get("r")
    if box.get("ok"):
        try:
            r.close()
        except Exception:  # noqa: BLE001
            pass
        return r
    if r is not None:
        try:
            r.close()                                 # unblock the stuck read
        except Exception:  # noqa: BLE001
            pass
    paths.log(f"GET {url[:120]} failed: {box.get('err') or 'wall-clock timeout'}")
    return None


def _status(engine: str, r) -> bool:
    if r is None or r.status_code != 200:
        paths.log(f"engine {engine} http {getattr(r, 'status_code', 'none')}")
        return False
    return True


def _clean_result_url(href: str) -> str:
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    u = urlparse(href)
    q = parse_qs(u.query)
    if "duckduckgo.com" in u.netloc and "uddg" in q:          # DDG redirect
        return unquote(q["uddg"][0])
    if "bing.com" in u.netloc and u.path.startswith("/ck/") and "u" in q:  # Bing click-tracker
        v = q["u"][0]
        if v.startswith("a1"):
            v = v[2:]
            try:
                return base64.urlsafe_b64decode(v + "=" * (-len(v) % 4)).decode("utf-8", "ignore")
            except Exception:  # noqa: BLE001
                return ""
    if u.scheme not in ("http", "https"):
        return ""
    return href


def _ddg(query: str, timeout: float) -> list[Hit]:
    # Up to three attempts (POST html, GET html, POST lite) share ONE deadline, so a
    # stalling DuckDuckGo costs `timeout` seconds in total, not three times that.
    end = time.time() + timeout

    def left() -> float:
        return end - time.time()

    def good(resp) -> bool:
        return resp is not None and resp.status_code == 200 and "result__a" in resp.text

    r = None
    real = DDG_URL.startswith("https://html.duckduckgo.com")
    if "{q}" in DDG_URL and real:
        r = _get("https://html.duckduckgo.com/html/", left(), "POST", data={"q": query, "kl": "us-en"})
    if not good(r) and left() > 0.5:
        r = _get(DDG_URL.format(q=quote_plus(query)), left())
    if not good(r) and real and left() > 0.5:
        lite = _get("https://lite.duckduckgo.com/lite/", left(), "POST", data={"q": query, "kl": "us-en"})
        if lite is not None and lite.status_code == 200:
            soup = BeautifulSoup(lite.text, "html.parser")
            out = []
            for a in soup.select("a.result-link"):
                tr = a.find_parent("tr")
                sn = tr.find_next_sibling("tr") if tr else None
                snip = sn.select_one(".result-snippet") if sn else None
                out.append(Hit(_clean_result_url(a.get("href", "")), a.get_text(" ", strip=True),
                               snip.get_text(" ", strip=True) if snip else "", "ddg"))
            return out
    if r is None or r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    hits = []
    for res in soup.select(".result"):
        a = res.select_one("a.result__a")
        if not a or "result--ad" in (res.get("class") or []):
            continue
        sn = res.select_one(".result__snippet")
        hits.append(Hit(_clean_result_url(a.get("href", "")), a.get_text(" ", strip=True),
                        sn.get_text(" ", strip=True) if sn else "", "ddg"))
    return hits


def _bing(query: str, timeout: float) -> list[Hit]:
    # %20, not "+": with "+" Bing's bot-facing page matched only the first word.
    r = _get(BING_URL.format(q=quote(query, safe="")), timeout)
    if not _status("bing", r):
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    hits = []
    for li in soup.select("li.b_algo"):
        a = li.select_one("h2 a")
        if not a:
            continue
        sn = li.select_one(".b_caption p") or li.select_one("p")
        hits.append(Hit(_clean_result_url(a.get("href", "")), a.get_text(" ", strip=True),
                        sn.get_text(" ", strip=True) if sn else "", "bing"))
    return hits


YAHOO_URL = os.environ.get("MC4_YAHOO_URL", "https://search.yahoo.com/search?p={q}&ei=UTF-8")


def _yahoo(query: str, timeout: float) -> list[Hit]:
    end = time.time() + timeout
    url = YAHOO_URL.format(q=quote(query, safe=""))
    r = _get(url, timeout)
    if r is not None and r.status_code >= 500 and end - time.time() > 1.5:
        time.sleep(0.4)                               # one retry on a 5xx, inside the same deadline
        r = _get(url, end - time.time())
    if not _status("yahoo", r):
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    hits = []
    for div in soup.select("div.algo"):
        a = div.select_one("h3 a") or div.select_one("a")
        if not a:
            continue
        href = a.get("href", "")
        m = re.search(r"/RU=([^/]+)/", href)
        if m:
            href = unquote(m.group(1))
        title = div.select_one("h3")
        sn = div.select_one(".compText") or div.select_one("p")
        hits.append(Hit(_clean_result_url(href), (title or a).get_text(" ", strip=True),
                        sn.get_text(" ", strip=True) if sn else "", "yahoo"))
    return hits


def _mojeek(query: str, timeout: float) -> list[Hit]:
    r = _get(MOJEEK_URL.format(q=quote_plus(query)), timeout)
    if r is None or r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    hits = []
    for li in soup.select("ul.results-standard li"):
        a = li.select_one("a.title") or li.select_one("h2 a") or li.select_one("a")
        if not a:
            continue
        sn = li.select_one("p.s")
        hits.append(Hit(_clean_result_url(a.get("href", "")), a.get_text(" ", strip=True),
                        sn.get_text(" ", strip=True) if sn else "", "mojeek"))
    return hits


def _retry_after(r) -> float:
    try:
        return float(r.headers.get("retry-after", ""))
    except (TypeError, ValueError):
        return 0.5


def _wiki(query: str, timeout: float) -> list[Hit]:
    """Wikipedia search. The attempts share ONE deadline: the action API, then the
    REST search endpoint, then the ordinary search page (separate routes, so a
    throttled API does not end the search), then the action API once more if the
    server's Retry-After fits."""
    params = {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 4}
    end = time.time() + timeout
    base = WIKI_API.rsplit("/w/api.php", 1)[0]

    def left() -> float:
        return end - time.time()

    def page_url(title: str) -> str:
        return f"{base}/wiki/{quote_plus(title.replace(' ', '_'), safe='_()')}"

    def text(html: str) -> str:
        return BeautifulSoup(html or "", "html.parser").get_text(" ", strip=True)

    def from_api(r) -> list[Hit] | None:
        if r is None or r.status_code != 200:
            return None
        try:
            items = r.json().get("query", {}).get("search", [])
        except (ValueError, AttributeError):
            return None
        return [Hit(page_url(it.get("title", "")), it.get("title", ""), text(it.get("snippet")), "wiki") for it in items]

    def from_rest(r) -> list[Hit] | None:
        if r is None or r.status_code != 200:
            return None
        try:
            items = r.json().get("pages", [])
        except (ValueError, AttributeError):
            return None
        return [Hit(page_url(it.get("key") or it.get("title", "")), it.get("title", ""),
                    text(it.get("excerpt")) or (it.get("description") or ""), "wiki") for it in items]

    def from_html(r) -> list[Hit] | None:
        if r is None or r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        out = []
        for li in soup.select("li.mw-search-result")[:4]:
            a = li.select_one(".mw-search-result-heading a")
            if not a or not a.get("href", "").startswith("/wiki/"):
                continue
            sn = li.select_one(".searchresult")
            out.append(Hit(base + a["href"], a.get("title") or a.get_text(" ", strip=True),
                           sn.get_text(" ", strip=True) if sn else "", "wiki"))
        return out or None                            # an empty page may be a block page: keep trying

    url = WIKI_API + "?" + "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items())
    r = _get(url, timeout)
    hits = from_api(r)
    if hits is not None:
        return hits
    paths.log(f"engine wiki http {getattr(r, 'status_code', 'none')}; trying the REST search route")
    wait = _retry_after(r) if r is not None and r.status_code == 429 else None
    if left() > 1.0:
        hits = from_rest(_get(f"{base}/w/rest.php/v1/search/page?q={quote_plus(query)}&limit=4", left()))
        if hits is not None:
            return hits
    if left() > 1.0:
        hits = from_html(_get(f"{base}/w/index.php?search={quote_plus(query)}&ns0=1&fulltext=1", left()))
        if hits is not None:
            return hits
    if wait is not None and left() - wait > 1.0:
        time.sleep(wait)
        hits = from_api(_get(url, left()))
        if hits is not None:
            return hits
    paths.log("engine wiki: no route answered")
    return []


ENGINE_WEIGHT = {"wiki": 1.5, "ddg": 1.2, "yahoo": 1.1, "bing": 1.0, "mojeek": 0.9}
_ENGINE_FN = {"ddg": _ddg, "bing": _bing, "wiki": _wiki, "mojeek": _mojeek, "yahoo": _yahoo}


# An engine that is unreachable from the grader's network would make every search
# wait for the full timeout. After RESTING_AFTER searches in a row without a single
# hit, an engine is rested; it is tried again every RETRY_EVERY-th search so it can
# come back. At least two engines always stay in use. Wikipedia is never rested: when
# it is throttled it says so at once, which costs no time.
RESTING_AFTER, RETRY_EVERY = 3, 6
NEVER_REST = ("wiki",)
_misses: dict[str, int] = {}
_searches = 0
_loaded = False
_STATE = paths.WORK_DIR / "engines.json"              # app.py may be started once per question


def _load_state() -> None:
    global _loaded, _searches
    if _loaded:
        return
    _loaded = True
    try:
        d = json.loads(_STATE.read_text(encoding="utf-8"))
        _searches = int(d.get("searches", 0))
        _misses.update({str(k): int(v) for k, v in dict(d.get("misses", {})).items()})
    except (OSError, ValueError, TypeError, AttributeError):
        pass


def _save_state() -> None:
    try:
        _STATE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _STATE.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"searches": _searches, "misses": _misses}), encoding="utf-8")
        os.replace(tmp, _STATE)
    except OSError:
        pass


def _engines_now() -> list[str]:
    all_e = [e for e in ENGINES if e in _ENGINE_FN]
    if _searches % RETRY_EVERY == 0:
        return all_e
    live = [e for e in all_e if e in NEVER_REST or _misses.get(e, 0) < RESTING_AFTER]
    return live if len(live) >= 2 else all_e


def search(queries: list[str], timeout: float) -> list[Hit]:
    """Query every engine with every query in parallel; fuse by reciprocal rank."""
    from .rank import terms
    global _searches
    _load_state()
    _searches += 1
    engines = _engines_now()
    got: dict[str, int] = {e: 0 for e in engines}
    jobs = []
    for q in queries:
        for e in engines:
            # Bing and Mojeek answer bot traffic poorly for question-form queries
            # (Bing collapses them to one word): send them keywords only.
            if q.startswith("site:"):
                qq = q
            elif e == "ddg":
                qq = q
            elif e == "wiki":
                qq = re.sub(r"[?*\"~]", " ", q).strip()          # ? and * are wildcards in CirrusSearch
            else:
                qq = " ".join(terms(q)[:10]) or q
            if (e, qq) not in jobs:
                jobs.append((e, qq))
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}
    if not jobs:
        return []
    # A throttled engine sometimes answers a query with results for its first word
    # alone ("wrote ..." -> dictionary pages). A hit that shares fewer than two of
    # the question's terms is kept, but ranked below every on-topic hit.
    qterms = [t for t in terms(queries[0]) if not t.startswith("http")]
    need = min(2, len(qterms))

    def on_topic(h: Hit) -> bool:
        text = f"{h.title} {h.snippet} {unquote(h.url)}".lower()
        return sum(1 for t in qterms if t in text or (len(t) > 4 and t[:-1] in text)) >= need
    # No `with`: leaving a with-block waits for every engine thread, so one stalled
    # engine would hold the whole search. Stragglers are abandoned instead (as in
    # fetch_many); their requests are wall-clock capped and the process exits hard.
    ex = ThreadPoolExecutor(max_workers=len(jobs))
    futs = {ex.submit(_ENGINE_FN[e], q, timeout): (e, q) for e, q in jobs}
    try:
        for f in as_completed(futs, timeout=timeout + 1.0):
            try:
                res = f.result()
            except Exception as e:  # noqa: BLE001
                paths.log(f"engine {futs[f]} error {e}")
                continue
            paths.log(f"engine {futs[f][0]} -> {len(res)} hits")
            got[futs[f][0]] += len(res)
            for i, h in enumerate(res[:10]):
                if not h.url:
                    continue
                key = _norm_url(h.url)
                w = ENGINE_WEIGHT.get(futs[f][0], 1.0)
                gain = w / (2 + i) + (0.5 if i < 2 else 0.0)   # top-2 of every engine always surface
                scores[key] = scores.get(key, 0.0) + (gain if on_topic(h) else gain * 0.05)
                if key not in best or len(h.snippet) > len(best[key].snippet):
                    h.rank = i
                    best[key] = h
    except Exception:  # noqa: BLE001 - as_completed timeout: keep what we have
        paths.log("search timed out; using partial results")
    ex.shutdown(wait=False, cancel_futures=True)
    for e, n in got.items():
        _misses[e] = 0 if n else _misses.get(e, 0) + 1
        if _misses[e] == RESTING_AFTER and e not in NEVER_REST:
            paths.log(f"engine {e}: no hits in {RESTING_AFTER} searches, resting it")
    _save_state()
    order = sorted(scores, key=lambda k: -scores[k])
    return [best[k] for k in order]


def _norm_url(u: str) -> str:
    p = urlparse(u)
    return (p.netloc.lower().removeprefix("www.") + p.path.rstrip("/") + ("?" + p.query if p.query else "")).lower()


# --------------------------------------------------------------------------- pages

_DROP = ["script", "style", "noscript", "svg", "template", "iframe", "form", "button", "select", "input"]
_BOILER = ["nav", "footer", "aside"]
_BOILER_CLASS = re.compile(r"(^|[-_ ])(nav|navbar|menu|footer|cookie|banner|breadcrumb|sidebar|navbox|mw-jump|reflist|references)([-_ ]|$)", re.I)


def _table_lines(table) -> list[str]:
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append((tr, cells))
    if not rows:
        return []
    out = []
    header = None
    first_tr, first = rows[0]
    if first_tr.find("th") and not first_tr.find("td") and len(first) > 1:
        header = first
        rows = rows[1:]
    for tr, cells in rows:
        if header and len(header) == len(cells):
            out.append("; ".join(f"{h}: {c}" for h, c in zip(header, cells)))
        elif len(cells) == 2:
            out.append(f"{cells[0]}: {cells[1]}")          # infobox style
        else:
            out.append(" | ".join(cells))
    return out


def html_to_page(url: str, html: str) -> Page:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        absu = urljoin(url, href).split("#", 1)[0]
        if absu.startswith(("http://", "https://")):
            links.append((absu, a.get_text(" ", strip=True)[:120]))
    for t in soup(_DROP):
        t.decompose()
    for t in soup(_BOILER):
        t.decompose()
    for t in soup.find_all(True, attrs={"class": True}):
        if t.attrs is None:
            continue
        cls = " ".join(t.get("class") or [])
        if _BOILER_CLASS.search(cls) and t.name not in ("body", "main", "article", "table"):
            t.decompose()
    for sup in soup.find_all("sup", class_="reference"):
        sup.decompose()
    for table in soup.find_all("table"):
        lines = _table_lines(table)
        table.replace_with(soup.new_string("\n" + "\n".join(lines) + "\n"))
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for blk in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "dt", "dd", "tr", "div", "section", "pre", "blockquote"]):
        blk.insert_before("\n")
        blk.insert_after("\n")
    root = soup.body or soup
    raw = root.get_text(" ")
    lines = []
    for ln in raw.split("\n"):
        ln = re.sub(r"[ \t ]+", " ", ln).strip()
        if ln:
            lines.append(ln)
    return Page(url=url, title=title, lines=lines, links=links, ok=bool(lines))


def pdf_to_page(url: str, data: bytes) -> Page:
    try:
        from pypdf import PdfReader
        rd = PdfReader(io.BytesIO(data))
        lines = []
        for pg in rd.pages[:30]:
            for ln in (pg.extract_text() or "").split("\n"):
                ln = ln.strip()
                if ln:
                    lines.append(ln)
        return Page(url=url, title=url.rsplit("/", 1)[-1], lines=lines, ok=bool(lines))
    except Exception as e:  # noqa: BLE001
        paths.log(f"pdf {url}: {e}")
        return Page(url=url)


_WIKI_PAGE = re.compile(r"^https?://([a-z\-]+)\.(?:m\.)?wikipedia\.org/wiki/([^?#]+)")


def _fetch_wiki(url: str, m, timeout: float) -> Page:
    api = f"https://{m.group(1)}.wikipedia.org/w/api.php"
    r = _get(f"{api}?action=parse&format=json&formatversion=2&redirects=1&prop=text|displaytitle&page={m.group(2)}", timeout)
    try:
        d = r.json()["parse"] if r is not None and r.status_code == 200 else None
    except (ValueError, KeyError):
        d = None
    if not d:
        return Page(url=url)
    p = html_to_page(url, "<html><head><title>" + BeautifulSoup(d.get("displaytitle", ""), "html.parser").get_text()
                     + " - Wikipedia</title></head><body>" + d.get("text", "") + "</body></html>")
    return p


def fetch(url: str, timeout: float) -> Page:
    m = _WIKI_PAGE.match(url)
    end = time.time() + timeout
    if m and not WIKI_API.startswith("http://127."):
        # The parse API gives the cleanest text but is the first thing Wikipedia
        # throttles; cap it so the ordinary page still fits in the same budget.
        p = _fetch_wiki(url, m, max(1.0, timeout * 0.6))
        if p.ok:
            return p
        paths.log(f"wiki parse route failed for {url[:100]}; fetching the page itself")
    r = _get(url, max(1.0, end - time.time()))
    if r is None or r.status_code >= 400:
        return Page(url=url)
    ctype = (r.headers.get("content-type") or "").lower()
    final = r.url or url
    if "pdf" in ctype or r.content[:5] == b"%PDF-":
        return pdf_to_page(final, r.content)
    if "json" in ctype or "text/plain" in ctype:
        text = r.content.decode(r.encoding or "utf-8", "ignore")
        return Page(url=final, title=final, lines=[ln.strip() for ln in text.splitlines() if ln.strip()][:4000], ok=True)
    if "html" in ctype or "xml" in ctype or not ctype:
        if r.encoding is None or r.encoding.lower() == "iso-8859-1":
            r.encoding = r.apparent_encoding or "utf-8"
        try:
            return html_to_page(final, r.text)
        except Exception as e:  # noqa: BLE001
            paths.log(f"parse {final}: {e}")
    return Page(url=final)


def fetch_many(urls: list[str], deadline: float) -> list[Page]:
    urls = list(dict.fromkeys(u for u in urls if u))
    if not urls:
        return []
    budget = max(1.0, deadline - time.time())
    out: dict[str, Page] = {}
    ex = ThreadPoolExecutor(max_workers=min(12, len(urls)))
    futs = {ex.submit(fetch, u, min(8.0, budget)): u for u in urls}
    try:
        for f in as_completed(futs, timeout=budget):
            try:
                out[futs[f]] = f.result()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001 - deadline: keep what arrived
        paths.log(f"fetch deadline hit: {len(out)}/{len(urls)} pages")
    ex.shutdown(wait=False, cancel_futures=True)
    return [out[u] for u in urls if u in out and out[u].ok]
