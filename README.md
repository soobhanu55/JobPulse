# JobPulse

What do German AI / ML / data student jobs actually ask for, and how much German do they really need?

I built this while applying for AI working student roles myself. Job boards show one posting at a time; I wanted the whole picture, from my own dataset instead of a clean Kaggle file.

![Skills asked in German AI/ML/Data student postings](skills.png)

## Data

- **Source:** the public job search API of the Bundesagentur für Arbeit (the one behind arbeitsagentur.de). LinkedIn, Indeed and StepStone forbid scraping in their terms, so they are not used.
- **Collected:** 519 postings from 10 searches ("Werkstudent KI", "Praktikum Machine Learning", ...), 26 Sep 2026, rate-limited, raw JSON cached per posting.
- **Cleaned:** search terms like "Werkstudent KI" also return sales and controlling jobs, so only postings whose title is really an AI/data role are kept: **184 of 519**. Stubs that only link to an external page are dropped.

## Findings

Full numbers in [`results.md`](results.md).

- **Python (40%) and SQL (26%) are the baseline.** LLM / GenAI is already in a quarter of postings, level with SQL.
- **Business tools matter more than ML frameworks:** Power BI (19%) and Excel (19%) are asked far more often than PyTorch (3%) or TensorFlow (1%). Many "AI" student jobs are about applying AI in a business, not training models.
- **German:** 41% ask for fluent (C1+) German and another 14% for good German. Only 3% call German optional. The 43% that don't mention it are all written in German, so German is usually implied there too.
- **Home office** is possible in only 16% of postings, and München (30) has more postings than Berlin and Hamburg combined (21).

## Can you tell from the title whether fluent German is required?

A practical question when you can't read hundreds of descriptions: does the title, company and city predict the German requirement?

| Model (5-fold CV) | F1 | Accuracy |
|---|---|---|
| Random guess at the class rate | 0.476 | - |
| Always "not fluent" | 0.000 | 0.592 |
| TF-IDF + logistic regression | 0.529 | 0.587 |

**Honest answer: no.** The model is only slightly better than guessing on F1 and not better than the majority baseline on accuracy. The German requirement lives in the description, not in the title, so there is no shortcut: you have to read it.

## Limits

- **Sampling bias:** all 184 relevant postings are in German. English-first employers mostly post on LinkedIn, which isn't scraped, so the German shares are an upper bound for the whole market.
- **Labels are rules, not people:** skills and German level come from regular expressions. They are spot-checked on real postings and covered by [`test_rules.py`](test_rules.py), but not hand-labelled at scale.
- **One snapshot:** a single day in September 2026.

## Run it

```bash
pip install requests pandas scikit-learn matplotlib
python scrape.py      # ~10 min, writes data/postings.csv (not committed: the ad texts belong to the employers)
python analyze.py     # writes results.md and skills.png
python test_rules.py  # checks the German-level rules
```
