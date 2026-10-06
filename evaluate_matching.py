"""Proxy evaluation of the three rankers.

There are no real "this CV fits this job" labels, so this does NOT measure CV-to-job fit. It measures
something narrower that a good matcher must get right: given one posting's description as the query,
are the nearest other postings of the same ROLE TYPE (genai/agents, data engineering, BI/analytics,
ML/data science)? Role type is labelled from the posting TITLE by keyword rules, and the text used for
ranking excludes the title, so the rankers cannot just match title words. Postings with a generic "KI/AI"
title carry no specific role type and are used only as distractors in the pool, never as queries.

    python evaluate_matching.py     # writes matching_results.md
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from analyze import load_relevant
from match import Matcher

FAMILIES = {  # checked in this order
    "genai_agents": r"llm|genai|generativ|agent|prompt|copilot|\brag\b",
    "data_engineering": r"data engineer|dateningenieur|datenplattform|data platform|\betl\b|big data|dateninfrastruktur",
    "bi_analytics": r"analytics|analyst|\bbi\b|reporting|controlling|business intelligence|power ?bi",
    "ml_data_science": r"data scien|machine learning|\bml\b|deep learning|computer vision|\bnlp\b|bildverarbeitung|scientific",
}
METHODS = ["tfidf", "embedding", "hybrid"]
K = 5


def family(title: str) -> str | None:
    t = title.lower()
    return next((name for name, pat in FAMILIES.items() if re.search(pat, t)), None)


def precision_and_rr(ranked_labels: list, query_label: str, k: int = K) -> tuple[float, float]:
    """Precision@k, and reciprocal rank of the first same-label result (0 if none in the list)."""
    hits = [lab == query_label for lab in ranked_labels]
    first = next((i + 1 for i, h in enumerate(hits) if h), None)
    return sum(hits[:k]) / k, (1 / first if first else 0.0)


def random_precision(labels: list, queries: list[int]) -> float:
    """Expected precision@k of ranking at random: the share of the other postings with the query's label."""
    n = len(labels)
    return float(np.mean([(sum(l == labels[q] for l in labels) - 1) / (n - 1) for q in queries]))


def bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(postings: pd.DataFrame, matcher: Matcher, methods: list[str] = METHODS, k: int = K) -> dict:
    labels = [family(t) for t in postings.title]
    queries = [i for i, lab in enumerate(labels) if lab]
    per_query = {m: [] for m in methods}
    for q in queries:
        for m in methods:
            s = matcher.scores(matcher.docs[q], m)
            order = [i for i in np.argsort(-s, kind="stable") if i != q]  # leave the query posting out
            per_query[m].append(precision_and_rr([labels[i] for i in order], labels[q], k))
    return {
        "n_queries": len(queries), "n_pool": len(postings), "random_p": random_precision(labels, queries),
        "families": pd.Series([labels[q] for q in queries]).value_counts().to_dict(),
        "methods": {m: np.array(v) for m, v in per_query.items()},
    }


def main() -> None:
    postings, _ = load_relevant()
    res = evaluate(postings, Matcher(postings, use_title=False))

    lines = [
        f"# Matching evaluation (proxy: same-role-type neighbours)\n",
        f"{res['n_queries']} query postings with a specific role type, ranked against a pool of {res['n_pool']} postings "
        f"(leave-one-out, description text only). Role types: {res['families']}.\n",
        f"| Ranker | Precision@{K} (95% CI) | MRR |", "|---|---|---|",
        f"| Random ranking | {res['random_p']:.3f} | - |",
    ]
    for m, arr in res["methods"].items():
        lo, hi = bootstrap_ci(arr[:, 0])
        lines.append(f"| {m} | {arr[:, 0].mean():.3f} ({lo:.3f} to {hi:.3f}) | {arr[:, 1].mean():.3f} |")
    lines.append(f"\nPaired differences in precision@{K} (same queries, 95% bootstrap CI):\n")
    for a, b in [("embedding", "tfidf"), ("hybrid", "tfidf"), ("hybrid", "embedding")]:
        d = res["methods"][a][:, 0] - res["methods"][b][:, 0]
        lo, hi = bootstrap_ci(d)
        lines.append(f"- {a} minus {b}: {d.mean():+.3f} ({lo:+.3f} to {hi:+.3f})")
    text = "\n".join(lines) + "\n"
    open("matching_results.md", "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
