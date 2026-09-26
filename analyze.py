"""Turn data/postings.csv into answers: which skills are asked, how much German is really required,
and whether the German requirement can be predicted without reading the full description.

    python analyze.py          # writes results.md and skills.png
"""
from __future__ import annotations

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

SKILLS = {
    "Python": r"\bpython\b", "SQL": r"\bsql\b", "PyTorch": r"\bpytorch\b", "TensorFlow": r"\btensorflow\b",
    "scikit-learn": r"scikit|sklearn", "LLM / GenAI": r"\bllms?\b|large language|generative ai|genai|generativ",
    "RAG": r"\brag\b|retrieval.augmented", "LangChain / LangGraph": r"langchain|langgraph", "Agents": r"\bagent(?:s|ic|en)?\b|ki-agent",
    "Docker": r"\bdocker\b", "Kubernetes": r"kubernetes|\bk8s\b", "Azure": r"\bazure\b", "AWS": r"\baws\b",
    "GCP": r"\bgcp\b|google cloud", "Databricks": r"databricks", "Spark": r"\bspark\b", "Power BI": r"power ?bi",
    "Tableau": r"tableau", "Excel": r"\bexcel\b", "Git": r"\bgit\b|github|gitlab", "Java": r"\bjava\b",
    "C++": r"c\+\+", "JavaScript / TypeScript": r"javascript|typescript", "SAP": r"\bsap\b",
    "n8n": r"\bn8n\b", "MCP": r"\bmcp\b|model context protocol", "Computer Vision": r"computer vision|bildverarbeitung",
    "NLP": r"\bnlp\b|natural language",
}
# German level asked for. "fluent" = C1+/very good/negotiation-level; checked in this order.
GERMAN_FLUENT = r"(sehr gute|fließende|fliessende|verhandlungssichere|exzellente|hervorragende)\w*\s+(deutsch|kenntnisse der deutschen)|deutsch\w*\s*\(?(c1|c2|muttersprach)|fluent\w*\s+(in\s+)?german|german\s*\(?(c1|c2|native|fluent)|(very good|excellent|business.fluent)\s+(command of\s+)?german|verhandlungssicher\w*\s+deutsch"
GERMAN_GOOD = r"gute\w*\s+deutsch|deutschkenntnisse|good\s+(command of\s+)?german|german\s+skills|deutsch\w*\s*\(?b[12]"
GERMAN_OPTIONAL = r"deutsch\w*[^.]{0,40}(von vorteil|wünschenswert|plus)|german[^.]{0,40}(is a plus|advantage|nice to have|beneficial)"
# Search terms like "Werkstudent KI" also return sales/controlling jobs; keep titles that are really AI/data roles.
RELEVANT_TITLE = r"\bki\b|\bki-|künstlich|\bai\b|machine learning|\bml\b|data|daten|\bllm|genai|generativ|analytics|analyst|deep learning|computer vision|\bnlp\b|agent|copilot|prompt"
EN_WORDS = r"\b(the|and|you|with|our|we|your|will|for)\b"
DE_WORDS = r"\b(und|die|der|du|mit|wir|deine|sie|für|ihre)\b"


def german_level(text: str) -> str:
    t = text.lower()
    if re.search(GERMAN_OPTIONAL, t):
        return "optional"
    if re.search(GERMAN_FLUENT, t):
        return "fluent (C1+)"
    if re.search(GERMAN_GOOD, t):
        return "good (B1-B2)"
    return "not mentioned"


def table(df: pd.DataFrame) -> str:
    """Markdown table without the tabulate dependency."""
    head = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
    rows = ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in zip(df.index, df.values)]
    return "\n".join([head, "|" + "---|" * (len(df.columns) + 1), *rows])


def main() -> None:
    df = pd.read_csv("data/postings.csv").fillna("")
    scraped = len(df)
    df = df[df.description.str.len() > 200]  # drop stubs that only link out
    df = df[df.title.str.lower().str.contains(RELEVANT_TITLE, regex=True)].copy()
    text = (df.title + " " + df.description).str.lower()
    df["lang"] = [("English" if len(re.findall(EN_WORDS, t)) > len(re.findall(DE_WORDS, t)) else "German") for t in text]
    df["german"] = df.description.map(german_level)
    for name, pat in SKILLS.items():
        df[name] = text.str.contains(pat, regex=True)

    n = len(df)
    skills = df[list(SKILLS)].mean().sort_values(ascending=False) * 100
    german_all = df.german.value_counts(normalize=True) * 100
    german_by_lang = pd.crosstab(df.lang, df.german, normalize="index") * 100
    remote = df.homeoffice.astype(str).str.lower().eq("true").mean() * 100
    cities = df.city.value_counts().head(10)

    # Predictor: can the title + company + city alone tell whether fluent German will be required?
    y = df.german.eq("fluent (C1+)")
    X = df.title + " | " + df.company + " | " + df.city + " | " + df.lang
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=1000, class_weight="balanced"))
    cv = min(5, int(y.sum()), int((~y).sum()))
    f1_model = cross_val_score(model, X, y, cv=cv, scoring="f1").mean()
    f1_base = cross_val_score(DummyClassifier(strategy="stratified", random_state=0), X, y, cv=cv, scoring="f1").mean()
    acc_model = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()
    acc_base = cross_val_score(DummyClassifier(strategy="most_frequent"), X, y, cv=cv, scoring="accuracy").mean()

    top = skills.head(15)
    plt.figure(figsize=(8, 5))
    top[::-1].plot.barh(color="#1F3864")
    plt.xlabel(f"% of {n} postings mentioning it")
    plt.title("Skills asked in German AI/ML/Data student postings")
    plt.tight_layout()
    plt.savefig("skills.png", dpi=150)

    fmt = lambda s: "\n".join(f"| {k} | {v:.1f}% |" for k, v in s.items())
    md = f"""# Results ({n} relevant AI/data postings out of {scraped} scraped, {pd.Timestamp.today():%Y-%m-%d})

## Skills asked (% of postings)
| Skill | Share |
|---|---|
{fmt(skills)}

## German requirement
| Level | Share |
|---|---|
{fmt(german_all)}

Posting language: {(df.lang == "English").sum()} of {n} are written in English. The Bundesagentur board is overwhelmingly German-language, so English-first employers (who post on LinkedIn) are under-represented here; the German share above is an upper bound for the whole market, not a neutral estimate.

Home office possible: **{remote:.1f}%** of postings.

## Top cities
{table(cities.to_frame('postings'))}

## Can the German requirement be predicted from title, company and city alone?
Target: posting asks for fluent (C1+) German ({y.mean()*100:.1f}% of postings). {cv}-fold cross-validation.

| Model | F1 | Accuracy |
|---|---|---|
| Random guess at the class rate | {f1_base:.3f} | - |
| Always "not fluent" | 0.000 | {acc_base:.3f} |
| TF-IDF + logistic regression | {f1_model:.3f} | {acc_model:.3f} |
"""
    open("results.md", "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()
