"""Test double for the LLM (MC4_FAKE_LLM=1), used only by the offline tests.

It knows the expected answers of tests/questions.json but only answers when the
answer text really is inside the sources it was shown, so it measures search,
crawl, parsing, ranking and grounding without a GPU. Never the shipped path.
"""
from __future__ import annotations

import json
import os
import re


class FakeLLM:
    def __init__(self):
        path = os.environ.get("MC4_FAKE_ORACLE", "")
        self.qs = json.load(open(path, encoding="utf-8"))["questions"] if path else []

    def chat(self, system, text, max_new_tokens=120):
        if "QUESTION:" not in text or "SOURCES:" not in text:
            return text.rsplit("QUESTION:", 1)[-1].strip()[:80]          # reformulation: echo
        q = text.split("QUESTION:", 1)[1].split("\n\nSOURCES:", 1)[0].strip()
        blocks = re.split(r"\n\n(?=\[S\d+\] )", text.split("SOURCES:\n", 1)[1].split("\n\nRules:", 1)[0])
        for item in self.qs:
            if item["question"] != q:
                continue
            for ans in [item["answer"]] + item.get("aliases", []):
                for b in blocks:
                    m = re.match(r"\[S(\d+)\]", b)
                    if m and ans.lower() in b.lower():
                        return json.dumps({"found": True, "answer": ans, "source": f"S{m.group(1)}", "quote": ans})
        return json.dumps({"found": False})
