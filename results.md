# Results (184 relevant AI/data postings out of 519 scraped, 2026-09-26)

## Skills asked (% of postings)
| Skill | Share |
|---|---|
| Python | 39.7% |
| SQL | 25.5% |
| LLM / GenAI | 25.5% |
| Power BI | 19.0% |
| Excel | 18.5% |
| Agents | 17.9% |
| Azure | 8.7% |
| JavaScript / TypeScript | 8.2% |
| Databricks | 7.6% |
| Git | 7.6% |
| SAP | 7.1% |
| Tableau | 4.9% |
| n8n | 4.3% |
| RAG | 3.8% |
| AWS | 3.3% |
| Java | 2.7% |
| LangChain / LangGraph | 2.7% |
| scikit-learn | 2.2% |
| C++ | 2.2% |
| Docker | 2.2% |
| MCP | 2.2% |
| GCP | 1.6% |
| PyTorch | 1.6% |
| Spark | 1.6% |
| Computer Vision | 1.6% |
| Kubernetes | 1.1% |
| NLP | 1.1% |
| TensorFlow | 0.5% |

## German requirement
| Level | Share |
|---|---|
| not mentioned | 42.9% |
| fluent (C1+) | 40.8% |
| good (B1-B2) | 13.6% |
| optional | 2.7% |

Posting language: 0 of 184 are written in English. The Bundesagentur board is overwhelmingly German-language, so English-first employers (who post on LinkedIn) are under-represented here; the German share above is an upper bound for the whole market, not a neutral estimate.

Home office possible: **15.8%** of postings.

## Top cities
| city | postings |
|---|---|
| München | 30 |
| Berlin | 12 |
| Hamburg | 9 |
| Stuttgart | 7 |
| Düsseldorf | 7 |
| Bremen | 6 |
| Sindelfingen | 6 |
| Augsburg, Bayern | 5 |
| Bielefeld | 3 |
| Karlsruhe, Baden | 3 |

## Can the German requirement be predicted from title, company and city alone?
Target: posting asks for fluent (C1+) German (40.8% of postings). 5-fold cross-validation.

| Model | F1 | Accuracy |
|---|---|---|
| Random guess at the class rate | 0.476 | - |
| Always "not fluent" | 0.000 | 0.592 |
| TF-IDF + logistic regression | 0.529 | 0.587 |
