import re
import zlib

import numpy as np
import pandas as pd
import pytest

from evaluate_matching import evaluate, family, precision_and_rr, random_precision
from match import Matcher, chunk_words, rrf, skills_in


class FakeEmbedder:
    """Deterministic bag-of-words hashing vectors, so tests never download a model."""

    def encode(self, texts):
        out = np.zeros((len(texts), 64))
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                out[i, zlib.crc32(w.encode()) % 64] += 1
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


ML = "python pytorch machine learning neural networks model training deep learning research"
BI = "excel power bi dashboards reporting controlling kpi business analyst sql"


def postings(extra=None):
    rows = [(f"Werkstudent Machine Learning {i}", f"{ML} project {i} docker") for i in range(4)]
    rows += [(f"Werkstudent Data Analyst {i}", f"{BI} team {i}") for i in range(4)]
    rows += extra or []
    return pd.DataFrame(rows, columns=["title", "description"]).assign(company="C", city="X")


# ---- helpers ---------------------------------------------------------------

def test_chunk_words_short_text_is_one_chunk():
    assert chunk_words("a b c", size=10) == ["a b c"]


def test_chunk_words_overlaps_and_covers_everything():
    words = [f"w{i}" for i in range(250)]
    chunks = chunk_words(" ".join(words), size=100, overlap=20)
    assert [len(c.split()) for c in chunks][:2] == [100, 100]
    assert chunks[0].split()[80:] == chunks[1].split()[:20]  # 20-word overlap
    assert set(" ".join(chunks).split()) == set(words)       # nothing dropped


def test_rrf_hand_calculated():
    # item 0 is ranked 1st by A and 3rd by B; item 1 is 2nd in both; item 2 is 3rd by A and 1st by B
    fused = rrf([np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0])])
    assert fused[0] == pytest.approx(1 / 61 + 1 / 63)
    assert fused[1] == pytest.approx(2 / 62)
    assert fused[0] == pytest.approx(fused[2]) and fused[0] > fused[1]


def test_skills_in_uses_word_boundaries():
    assert skills_in("Python, SQL and Docker") == {"Python", "SQL", "Docker"}
    assert "Java" not in skills_in("We use JavaScript and TypeScript")
    assert "SQL" not in skills_in("SQLAlchemy models")


# ---- matcher ---------------------------------------------------------------

def test_tfidf_ranks_the_matching_role_family_first():
    m = Matcher(postings(), embedder=FakeEmbedder())
    top = m.rank("I know python pytorch and deep learning, neural networks", method="tfidf", k=4)
    assert all("Machine Learning" in t for t in top.title)
    top = m.rank("excel power bi reporting dashboards", method="tfidf", k=4)
    assert all("Data Analyst" in t for t in top.title)


def test_skill_gap_is_posting_skills_minus_cv_skills():
    m = Matcher(postings(), embedder=FakeEmbedder())
    top = m.rank("python pytorch deep learning neural networks", method="tfidf", k=1)
    assert "Python" in top.has[0] and "Docker" in top.gap[0] and "Python" not in top.gap[0]


def test_german_filter_hides_postings_that_need_fluent_german():
    extra = [("Werkstudent Machine Learning X", f"{ML} Sehr gute Deutschkenntnisse erforderlich")]
    m = Matcher(postings(extra), embedder=FakeEmbedder())
    cv = "python pytorch deep learning neural networks machine learning"
    assert any(g == "fluent (C1+)" for g in m.rank(cv, "tfidf", k=10).german)
    assert not any(g == "fluent (C1+)" for g in m.rank(cv, "tfidf", k=10, max_german="good (B1-B2)").german)


def test_hybrid_and_embedding_return_sorted_scores():
    m = Matcher(postings(), embedder=FakeEmbedder())
    for method in ("embedding", "hybrid"):
        s = m.rank("python deep learning neural networks", method=method, k=5).score.tolist()
        assert len(s) == 5 and s == sorted(s, reverse=True)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        Matcher(postings(), embedder=FakeEmbedder()).scores("x", "magic")


# ---- evaluation ------------------------------------------------------------

def test_precision_and_rr_by_hand():
    assert precision_and_rr(["a", "b", "a", "c"], "a", k=3) == (pytest.approx(2 / 3), 1.0)
    assert precision_and_rr(["b", "b", "a"], "a", k=2) == (0.0, pytest.approx(1 / 3))
    assert precision_and_rr(["b", "b"], "a", k=2) == (0.0, 0.0)


def test_random_precision_is_the_share_of_other_postings_with_that_label():
    assert random_precision(["a", "a", "b"], [0]) == pytest.approx(0.5)  # 1 of the 2 others shares the label


@pytest.mark.parametrize("title,fam", [
    ("Werkstudent Data Engineering (m/w/d)", "data_engineering"),
    ("Data Analyst Werkstudent", "bi_analytics"),
    ("Werkstudent LLM & Agents", "genai_agents"),
    ("Werkstudent Machine Learning", "ml_data_science"),
    ("Werkstudent Künstliche Intelligenz", None),
])
def test_family_rules(title, fam):
    assert family(title) == fam


def test_evaluate_finds_same_family_neighbours_with_a_sensible_matcher():
    df = postings()
    res = evaluate(df, Matcher(df, embedder=FakeEmbedder(), use_title=False), methods=["tfidf"], k=3)
    assert res["n_queries"] == 8
    assert res["methods"]["tfidf"][:, 0].mean() > res["random_p"] + 0.3
