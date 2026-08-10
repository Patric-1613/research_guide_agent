# Report quality evaluation fixtures (R6A/R6B)

Fixtures for the `report_quality` eval suite. See
`specs/report-quality-evaluation-plan.md` for the full frozen design
(result schema, hard-failure identifiers, informational signals, judge
strategy, R6D/pairwise design).

**R6B (deterministic/mock, complete)**: `research_agent/evals/
runners/run_report_quality.py` + `research_agent/evals/evaluators/
report_quality.py` implement all 6 frozen hard-failure checks plus
informational signals against these fixtures — offline, free,
independent of R4, no OpenAI calls. Run it:

```bash
uv run python -m research_agent.evals.cli run --suite report_quality
uv run python -m research_agent.evals.cli run --suite report_quality --mode mock --tags security
uv run python -m research_agent.evals.cli run --suite report_quality --mode mock --subset 3
```

`--mode live` is not implemented yet — it fails cleanly (exit 2, no
traceback, no CSV/detail side effects) until R6C adds the two bounded
live judge tasks (claim/source + holistic). See `tests/
test_evals_report_quality.py` for the harness's own test coverage.

## Layout

```
eval_data/report_quality/
  README.md          <- this file
  manifest.jsonl      <- one line per fixture: id, path, tags, source_origin, notes
  fixtures/*.json      <- the fixtures themselves
```

**Why a manifest + individual JSON files, not one JSONL file** (unlike
`eval_data/chat_web_relevance_redteam.jsonl`): a report-quality fixture
embeds full paper abstracts, full web snippets, and a full 8-section
generated report — realistically several KB per fixture, too large to
stay reviewable crammed onto a single JSONL line the way a short
chat/web-relevance case can be. The manifest keeps the same per-record
metadata role `_METADATA_KEYS`/`load_examples` already give a JSONL
record (`id`, `tags`, `source`/`source_origin`, `notes`) while letting
each fixture's actual content live in its own diffable file.

**The manifest owns `tags`.** Fixture JSON bodies do not repeat a
`tags` field — see `specs/report-quality-evaluation-plan.md` section 4
for why.

## Fixture index

| id | template | tags | tests |
|---|---|---|---|
| `good_foundational` | foundational | baseline, foundational, template_fit, groundedness, citation_correctness | A well-grounded, well-cited, well-synthesized report. No hard failures. |
| `good_analytical` | analytical | baseline, analytical, template_fit, synthesis_quality, citation_correctness | Same evidence as `good_foundational`, analytical depth. No hard failures. |
| `good_expert` | expert | baseline, expert, template_fit, analytical_quality, citation_correctness | Same evidence as the two above, expert depth/nuance. No hard failures. |
| `citation_and_grounding_failure` | analytical | citation_correctness, groundedness, adversarial | Wrong-source citation, broad-topic-only citation, fabricated statistic — structurally valid, semantically wrong. |
| `verbose_low_synthesis` | analytical | synthesis_quality, analytical_quality, coherence | Accurately cited but paper-by-paper listing, cross-section repetition, duplicate Gap Analysis / Future Research Directions. |
| `source_prompt_injection` | analytical | groundedness, citation_correctness, adversarial, prompt_injection, security | Injected instructions inside a paper abstract and a web snippet; the report shows the injection succeeding. |
| `evaluator_injection_in_report` | analytical | coherence, adversarial, prompt_injection, security | The report's own generated prose tries to instruct an evaluator to score it highly. |
| `structural_and_metadata_corruption` | analytical | structural_integrity, hard_failure, adversarial | Stacks all 6 frozen hard-failure identifiers plus missing source metadata, for isolated per-checker testing. |

The three `good_*` fixtures deliberately share the exact same three
synthetic papers and one web article, so template-depth differences
are the only variable between them — a reviewer can compare all three
against identical evidence. `citation_and_grounding_failure` and
`verbose_low_synthesis` reuse the same three papers again, specifically
so a reviewer can compare correct vs. flawed use of identical evidence
against the `good_analytical` baseline.

## Fixture evidence conventions

- All paper/web evidence is **synthetic and hand-written** — invented
  paper titles, authors, and abstracts; invented web articles. Nothing
  is copied from a real publication.
- Every manifest entry records `source_origin: "synthetic_handwritten"`
  — there is no other origin in this fixture set yet. A future R6E
  fixture drawn from a real, human-annotated session would use a
  different `source_origin` value and would carry real
  `human_annotations`, never conflated with these synthetic
  `expected.dimension_labels`.
- URLs use `example.com`/`papers.example.com`-style domains throughout.
- No binary report exports (PDF/DOCX) are committed — fixtures store
  the report *dict*, the same shape `research_agent/report.py`
  produces before export rendering.

## `expected.dimension_labels` vs. `human_annotations`

Every fixture's `expected.dimension_labels` is a **synthetic, hand-
constructed expectation** written by whoever built the fixture, with a
short rationale a human reviewer can verify against the fixture's own
evidence (the abstracts/snippets/report content in the same file).
**This is not human calibration data and must never be described as
ground truth.** `human_annotations` is the field reserved for real
R6E-collected annotations on real reports — it is an empty list on
every fixture in this set, since no such annotations exist yet.

## Report-dict shape

`generated_report` in every fixture uses the real, current stored
report-dict shape, confirmed against
`research_agent/curation_session.py::_serialize_report` and
`research_agent/report.py`'s section-building functions: one entry per
the 8 Analytical section keys plus the 3 legacy projection keys
(`findings`/`limitations`/`future_scope`, straight aliases of their
mapped Analytical section), each shaped `{"content", "cited_papers",
"cited_web_articles", "reference_numbers"}`; a top-level
`skipped_papers` list; a top-level `references` list; a top-level
`sections` list (`key`/`title`/`content`/`reference_numbers`); and
`report_template`. Inline `[N]` citation markers are written directly
into each section's `content` string at their real rendered position —
not stripped — matching what `_build_references_and_renumber` actually
produces.

## Known gap this fixture set surfaces

`source_prompt_injection.json` demonstrates that report generation
currently has **no independent defense** against instruction-like
content inside a paper abstract or web snippet —
`research_agent/qa.py`'s `_detect_retrieved_prompt_injection` guard is
wired only into the chat/web-relevance filtering path, never into
`research_agent/report.py`'s own prompt construction, and never into
anything touching paper abstracts at all. This is documented as
evaluation-surface context, not fixed here — see
`specs/report-quality-evaluation-plan.md` section 7. R6A makes no
change to `research_agent/`.

## R6D (pairwise refinement evaluation)

Not built yet, and no pairwise fixtures exist in this directory. The
future pairwise fixture schema (blinded A/B, swapped order, per-
dimension A/B/tie, positional disagreement) is documented in
`specs/report-quality-evaluation-plan.md` section 8.
