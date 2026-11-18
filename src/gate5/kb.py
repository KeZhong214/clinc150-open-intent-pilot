"""Fictional-bank policy knowledge base with local TF-IDF retrieval and citations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
KB_PATH = ROOT / "knowledge_base" / "fictional_bank" / "policies.jsonl"


@dataclass
class KBDoc:
    doc_id: str
    title: str
    version: str
    source_id: str
    category: str
    text: str
    citation: str
    tags: list[str]


class KBRetriever:
    def __init__(self, path: Path = KB_PATH):
        self.path = path
        self.docs = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            self.docs.append(
                KBDoc(
                    doc_id=raw["doc_id"],
                    title=raw["title"],
                    version=raw["version"],
                    source_id=raw["source_id"],
                    category=raw["category"],
                    text=raw["text"],
                    citation=raw["citation"],
                    tags=raw.get("tags", []),
                )
            )
        if len(self.docs) < 25:
            raise RuntimeError(f"Expected >=25 policy docs, got {len(self.docs)}")
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        corpus = [f"{d.title}. {d.text} {' '.join(d.tags)}" for d in self.docs]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        order = sims.argsort()[::-1][:k]
        out = []
        for i in order:
            d = self.docs[int(i)]
            out.append(
                {
                    "doc_id": d.doc_id,
                    "title": d.title,
                    "version": d.version,
                    "source_id": d.source_id,
                    "category": d.category,
                    "text": d.text,
                    "citation": d.citation,
                    "score": float(sims[int(i)]),
                }
            )
        return out

    def render(self, query: str, k: int = 2) -> tuple[str, list[dict]]:
        hits = self.search(query, k=k)
        if not hits or hits[0]["score"] <= 0:
            return "I do not have a matching fictional policy document. I’ll hand this to a human specialist.", []
        body = " ".join(f"[{h['citation']}] {h['text']}" for h in hits[:2])
        return body, hits[:2]
