# AMD AI League – Match 4: Intelligent Web Information Retrieval

An agent that reads a natural-language request, searches and navigates the web, and returns the exact
piece of information asked for, with the page that proves it. Packaged as a Docker image on the
mandated AMD ROCm base; the language model runs on the AMD GPU.

> **Status (2026-11-10):** AMD/lablab have not published the Match 4 contract (CLI, JSON keys, limits)
> and the submission form has no Match 4 field yet. The entry point below follows the Match 3 shape and
> accepts common variants. When the real contract appears, only `app/app.py` (arguments, output file
> name, JSON keys) should need changing.

## How it works

1. **Seeds.** A start URL (from `--url` or written inside the question) is crawled best-first on its own
   site: links are ranked by how well their text matches the question. In parallel, several search engines
   (DuckDuckGo, Bing, Wikipedia, Mojeek) are queried and their results fused, so one engine blocking us is
   not fatal.
2. **Read pages.** Up to 8 pages are fetched in parallel (HTML and PDF). Navigation, footers and cookie
   banners are dropped; tables become `label: value` lines so infoboxes and spec tables survive.
3. **Rank.** Text is cut into ~90-word chunks and ranked with BM25; the best ~1,500 words go to the model.
4. **Answer.** A small instruct model (Qwen3-4B-Instruct-2507, bf16) on the AMD GPU returns strict JSON:
   the shortest exact answer, the source it came from and a supporting quote.
5. **Ground.** The answer must appear on the cited page (or on another fetched page, which is then cited).
   If nothing contains it, the agent follows the best links one level deeper and searches again with a
   rewritten query, then gives an honest empty answer rather than a guess.

Every step shares one deadline (`MC4_DEADLINE_S`, default 25 s per question). The model is loaded once by
a resident server (`--index` warm-up, or the first question), so each question only does web work plus one
short request.

## Contract (assumed)

```bash
python3 /app/app.py --index                                   # optional warm-up: load the model
python3 /app/app.py --query-id query_01 --query "question" [--url https://start.page]
python3 /app/app.py --input /app/input/query_01.json          # {"query": "...", "url": "..."}
python3 /app/app.py --input /app/input/questions.json         # list, or {"queries": [...]}
```

Output, one file per question, written atomically, empty answer on any failure:
`/app/output/<query-id>_output.json` → `{"answer": "1889", "citations": ["https://…"], "confidence": 0.9}`.
Nothing is printed to stdout; logs go to `/tmp/mc4/app.log`.

## Test

```bash
pip install requests beautifulsoup4 pypdf
python3 tests/run_local.py        # offline: fake web + fake LLM, 7 questions + odd invocations
python3 tests/live_recall.py      # real web, no GPU: is the answer in what the model would read?
MC4_REAL_LLM=1 python3 tests/run_local.py   # on an AMD GPU box with weights in /models/llm
```

The fake web (`tests/fake_web.py`) covers an infobox table with decoy years in the menu and footer, a PDF,
numbers with thousands separators, an unindexed site reachable only by crawling from a start URL, a start
URL inside the question, a spec table with a rumour page that disagrees, and an unanswerable question.

## Build

`Dockerfile` builds on the mandated ROCm base for `linux/amd64`. In CI the same image is assembled by
appending two layers (app + Python packages, then the model weights) to the untouched base with `crane`,
because the base alone is ~29 GiB and a hosted runner cannot unpack it. No secrets are in the image, and
nothing is downloaded at startup. The published image reference is given privately in the lablab
submission, as the League rules require.

Built by Abe (Abrsh23) for the AMD AI League. MIT licensed.
