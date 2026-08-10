"""R6C.2: live LLM judges for the report_quality suite -- claim_source.py
(citation_correctness/groundedness, batched per-claim verdicts) and
holistic.py (synthesis_quality/analytical_quality/template_fit/
coherence/source_balance, one call over the full sanitized report).
Both are pure "given already-prepared, already-sanitized input, call
the model and return structured verdicts" modules -- neither builds
its own evidence registry, claim extraction, sampling, or injection
detection; that is entirely R6C.1's job
(research_agent/evals/report_quality_inputs.py), consumed here, not
duplicated. Neither judge is reachable from production report
generation/refinement, any API route, or the frontend -- both are only
ever invoked from research_agent/evals/runners/run_report_quality.py's
own opt-in `--mode live` path.
"""
