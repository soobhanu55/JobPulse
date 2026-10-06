# Matching evaluation (proxy: same-role-type neighbours)

69 query postings with a specific role type, ranked against a pool of 184 postings (leave-one-out, description text only). Role types: {'bi_analytics': 32, 'ml_data_science': 19, 'genai_agents': 11, 'data_engineering': 7}.

| Ranker | Precision@5 (95% CI) | MRR |
|---|---|---|
| Random ranking | 0.118 | - |
| tfidf | 0.191 (0.148 to 0.235) | 0.452 |
| embedding | 0.258 (0.206 to 0.310) | 0.534 |
| hybrid | 0.252 (0.200 to 0.304) | 0.503 |

Paired differences in precision@5 (same queries, 95% bootstrap CI):

- embedding minus tfidf: +0.067 (+0.014 to +0.122)
- hybrid minus tfidf: +0.061 (+0.017 to +0.110)
- hybrid minus embedding: -0.006 (-0.035 to +0.026)
