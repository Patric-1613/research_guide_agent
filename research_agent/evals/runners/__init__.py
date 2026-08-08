"""Per-suite experiment runners. Each run_<suite>.py exposes a
`run_experiment(mode, subset=None, tags=None)` callable the CLI
discovers by suite name -- see `_base.py` for the shared loading/
scoring/aggregation logic every runner builds on.
"""
