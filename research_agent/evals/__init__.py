"""E0/R7D.1: this project's own eval package, decided as a design-only
checkpoint (E0, docs/evaluation.md's "Planned evaluation architecture"
section) and first built out here for one suite (chat_relevance,
mock-only). Never imported by api_app/app.py or any router -- stays
inert at runtime, the same precedent ranking.py's own eval-only BM25/
hybrid modes already set for eval-only code living inside
research_agent/. `scripts/eval_retrieval.py`/`scripts/ragas_eval.py`
are untouched by this package's existence.
"""
