"""Rank the scraped postings for a CV, and show what skills each posting asks that the CV lacks.

Three rankers over the same postings:
  tfidf      word/bigram TF-IDF + cosine similarity (the CineMatch content-based approach)
  embedding  multilingual sentence embeddings of chunked text (the ResumeLens model)
  hybrid     reciprocal-rank fusion of the two (no measured gain over embedding alone, see matching_results.md)

    python match.py examples/sample_cv.txt --k 5 --method embedding --max-german "good (B1-B2)"
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from analyze import SKILLS, german_level, load_relevant

EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"  # handles German + English, runs locally
GERMAN_ORDER = ["not mentioned", "optional", "good (B1-B2)", "fluent (C1+)"]
RRF_K = 60  # standard reciprocal-rank-fusion constant


def chunk_words(text: str, size: int = 100, overlap: int = 20) -> list[str]:
    """Overlapping word windows. The embedding model only reads ~128 tokens, so long
    postings (median ~3,300 chars) must be split or most of the text is silently ignored."""
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words) - overlap, step)]


def rrf(score_lists: list[np.ndarray]) -> np.ndarray:
    """Reciprocal-rank fusion: sum 1/(RRF_K + rank) over rankers (rank 1 = best)."""
    fused = np.zeros(len(score_lists[0]))
    for scores in score_lists:
        ranks = np.empty(len(scores), dtype=int)
        ranks[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
        fused += 1.0 / (RRF_K + ranks)
    return fused


def skills_in(text: str) -> set[str]:
    t = text.lower()
    return {name for name, pat in SKILLS.items() if re.search(pat, t)}


class SentenceEmbedder:
    """Lazy wrapper so importing this module (and the TF-IDF path) never loads torch."""

    def __init__(self, name: str = EMBED_MODEL):
        self.name, self._model = name, None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name)
        return np.asarray(self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


class Matcher:
    def __init__(self, postings: pd.DataFrame, embedder=None, use_title: bool = True):
        self.df = postings.reset_index(drop=True)
        self.docs = [(f"{t}. {d}" if use_title else d) for t, d in zip(self.df.title, self.df.description)]
        self.embedder = embedder or SentenceEmbedder()
        self.tfidf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_df=0.9)
        self.X = self.tfidf.fit_transform(self.docs)
        self._post_emb: np.ndarray | None = None
        self.post_skills = [skills_in(d) for d in self.docs]
        self.post_german = [german_level(d) for d in self.df.description]

    def _embed_pooled(self, texts: list[str]) -> np.ndarray:
        """One unit vector per text: chunk, embed every chunk, mean-pool, re-normalise."""
        chunks = [chunk_words(t) for t in texts]
        flat = self.embedder.encode([c for cs in chunks for c in cs])
        out, i = [], 0
        for cs in chunks:
            v = flat[i:i + len(cs)].mean(axis=0)
            out.append(v / (np.linalg.norm(v) + 1e-12))
            i += len(cs)
        return np.vstack(out)

    def scores(self, cv_text: str, method: str = "embedding") -> np.ndarray:
        tf = (self.X @ self.tfidf.transform([cv_text]).T).toarray().ravel()
        if method == "tfidf":
            return tf
        if self._post_emb is None:
            self._post_emb = self._embed_pooled(self.docs)
        emb = self._post_emb @ self._embed_pooled([cv_text])[0]
        if method == "embedding":
            return emb
        if method == "hybrid":
            return rrf([tf, emb])
        raise ValueError(f"unknown method {method!r}")

    def rank(self, cv_text: str, method: str = "embedding", k: int = 10, max_german: str | None = None) -> pd.DataFrame:
        """Top-k postings. max_german drops postings whose German requirement is stricter than the
        given level (one of GERMAN_ORDER), e.g. 'good (B1-B2)' hides postings asking fluent German."""
        s = self.scores(cv_text, method)
        order = np.argsort(-s, kind="stable")
        if max_german:
            limit = GERMAN_ORDER.index(max_german)
            order = [i for i in order if GERMAN_ORDER.index(self.post_german[i]) <= limit]
        cv_sk = skills_in(cv_text)
        rows = [{
            "title": self.df.title[i], "company": self.df.company[i], "city": self.df.city[i],
            "score": float(s[i]), "german": self.post_german[i],
            "has": sorted(self.post_skills[i] & cv_sk), "gap": sorted(self.post_skills[i] - cv_sk),
        } for i in order[:k]]
        return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cv", help="path to a plain-text CV")
    ap.add_argument("--method", choices=["tfidf", "embedding", "hybrid"], default="embedding")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-german", choices=GERMAN_ORDER, default=None)
    args = ap.parse_args()

    postings, _ = load_relevant()
    postings = postings[~postings.title.str.contains(r"duales studium|ausbildung", case=False, regex=True)]  # degree programmes, not jobs
    cv_text = open(args.cv, encoding="utf-8").read()
    top = Matcher(postings).rank(cv_text, args.method, args.k, args.max_german)
    for i, r in top.iterrows():
        print(f"{i + 1:>2}. {r.title}  [{r.city}]  score={r.score:.3f}  german={r.german}")
        print(f"      you have: {', '.join(r.has) or '-'}   |   gap: {', '.join(r.gap) or '-'}")


if __name__ == "__main__":
    main()
