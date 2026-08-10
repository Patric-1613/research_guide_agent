# R6 — report quality evaluation: frozen R6A design, plus what R6B/R6C actually shipped

This document freezes the R6 result schema, hard-failure identifiers,
informational-signal list, fixture architecture, and R6B/R6C/R6D
scoring semantics, per the R6A checkpoint (§0-11 below, unmodified
historical record). **R6A itself added no executable code** — it was
a design/fixture-only phase; R6B and R6C are the phases that actually
built against this frozen design. §12-14 (added at the R6C.3
checkpoint, 2026-08-11) record what R6C actually shipped against the
§10 design, the aggregation semantics it settled on, and the residual
policy debt left open when R6C was closed — read those sections for
current status; §0-11 remain the original frozen record and are
preserved rather than rewritten. See `eval_data/report_quality/
README.md` for the fixture set this document governs, `docs/
evaluation.md`'s "R6A" through "R6C.3" sections for the workflow-doc
cross-reference, and `specs/backend-backlog.md`'s R6A/R6B/R6C entries
for status.

## 0. Approved architectural decisions (binding on R6B onward)

1. **R6 is independent from R4.** R6 never calls `generate_report`,
   `evaluate_report`, or `revise_report`, and never treats R4's
   `overall_score` as ground truth. R6 consumes already-produced report
   dictionaries and the source evidence (`selected_papers`,
   `approved_web_articles`) they were generated from — as stored
   fixtures in R6A/R6B, and (later, R6C+) as real captured session data.
2. **No overall quality score yet.** No weights are invented in R6A.
   Hard deterministic defects remain strict pass/fail. Any future
   continuous LLM-judge score is informational only until R6E produces
   real calibration evidence — a future score of `0.85` must never by
   itself mean a failed suite run.
3. **R6B mock mode evaluates deterministic properties only.** It must
   never fabricate an overall LLM quality score. A later, R6C-only
   fixture-controlled mocked judge output exists solely to test R6C's
   own plumbing (the harness can consume a judge response correctly) —
   it is not quality evidence and must never be reported as if it were.
4. **R6C uses two bounded judge tasks, not one giant call**: a local
   claim/source (citation-correctness + groundedness) judge over
   sampled claim-source pairs, and a separate holistic judge
   (synthesis, analytical quality, template fit, coherence, source
   balance) over the whole report. This split is decided now so R6B's
   fixtures and result shape don't have to be reworked when R6C lands.
5. **Synthetic expected labels are not human calibration.** Every
   fixture in this R6A set carries hand-constructed
   `expected.dimension_labels` with a rationale a human reviewer can
   verify against the fixture's own evidence. These are explicitly
   distinguished from `human_annotations` (real, R6E-collected labels
   on real reports) in both the fixture schema (§4) and everywhere this
   document or `docs/evaluation.md` describes them. No text in this
   repository should ever call a synthetic fixture label "ground
   truth" or "calibration data."

## 1. Frozen result schema (`schema_version: "r6a-v1"`)

```json
{
  "schema_version": "r6a-v1",
  "structural_integrity": {
    "status": "pass|fail",
    "checks": {},
    "hard_failures": []
  },
  "informational_signals": {
    "section_word_counts": {},
    "citation_density_by_section": {},
    "source_citation_counts": {},
    "skipped_paper_rate": null
  },
  "judge_dimensions": {
    "citation_correctness": {"label": "pass|fail|not_applicable|unknown", "score": null, "reasons": []},
    "groundedness": {"label": "pass|fail|not_applicable|unknown", "score": null, "reasons": []},
    "synthesis_quality": {"label": "pass|fail|not_applicable|unknown", "score": null, "reasons": []},
    "analytical_quality": {"label": "pass|fail|not_applicable|unknown", "score": null, "reasons": []},
    "template_fit": {"label": "pass|fail|not_applicable|unknown", "score": null, "reasons": []},
    "coherence": {"label": "pass|fail|not_applicable|unknown", "score": null, "reasons": []},
    "source_balance": {"label": "pass|fail|not_applicable|unknown", "score": null, "reasons": []}
  },
  "hard_failures": [],
  "warnings": [],
  "not_applicable": [],
  "judge_metadata": null
}
```

This shape deliberately separates three kinds of information that a
single blended score (R4's `overall_score`) collapses into one number:

- **`structural_integrity`** — objective, deterministic, pass/fail.
  Never influenced by an LLM. `hard_failures` here uses the stable
  identifiers frozen in §2 and is duplicated at the top level
  (`hard_failures`) for a caller that wants the flat list without
  walking into `structural_integrity` — both lists are always
  identical; the nested one is scoped context, the top-level one is
  the flat aggregate future tooling (e.g. `run_suite`) reads.
- **`informational_signals`** — deterministic, cheap, **never a gate**.
  Purely descriptive measurements (§3). No threshold is frozen for any
  of them in R6A.
- **`judge_dimensions`** — LLM-produced (or, in R6B mock mode,
  fixture-controlled stand-ins that exercise the same shape without
  claiming to be real judgments). Each entry is categorical first
  (`label`), with `score` reserved for a future 0–1 continuous value
  R6C will populate and R6E will calibrate — **`score` does not
  determine pass/fail before R6E's calibration exists.**

Field semantics, frozen:

- `label` is categorical: `"pass"`, `"fail"`, `"not_applicable"`, or
  `"unknown"`. `"unknown"` means no valid live judgment was obtained
  (judge call failed, timed out, or returned an unusable response) —
  it is **not** the same as `"not_applicable"` (the dimension
  genuinely doesn't apply to this report, e.g. `source_balance` when
  only one source exists total) and neither is ever silently averaged
  in as if it were a `0`.
- `score` will later hold a 0–1 informational float once R6C exists.
  It carries no pass/fail authority on its own until R6E calibrates it
  — see decision 2 above.
- `not_applicable` (both the per-dimension label and the top-level
  list) is always explicit, never inferred, and never averaged as
  zero.
- **Hard failures override positive stylistic judgments.** A report
  with any `structural_integrity.hard_failures` entry must never be
  reported as high quality on the strength of good `judge_dimensions`
  labels — this mirrors `evaluate_report`'s own existing rule that
  deterministic hard issues force `needs_revision=True` regardless of
  the LLM's opinion (`research_agent/report.py:1783-1784`).
- `source_balance` is informational/context-sensitive, not
  automatically a gate — a single-source report citing its one source
  well is not "unbalanced" (see §3's explicit statement on this).
- `judge_metadata` is `null` until a live judge actually runs (R6C);
  it will then carry `{"model", "prompt_version", "latency_ms",
  "error"}`, the same discipline R7E's `direct_relevance_cache`
  established for judge-call provenance.

## 2. Frozen hard-failure identifiers

Stable string identifiers, referenced by `structural_integrity.checks`
keys and by `structural_integrity.hard_failures`/top-level
`hard_failures` entries. R6B's deterministic checkers must use exactly
these identifiers so fixture expectations (`expected.hard_failures` in
every fixture file) stay meaningful across the R6A → R6B transition.

| Identifier | Meaning | Precedent |
|---|---|---|
| `missing_required_section` | A section key the active template defines is entirely absent from the report dict. | New check; R4's own `_deterministic_report_checks` folds this and the next row into one combined message — R6 splits them for isolated testing. |
| `empty_required_section` | A required section key is present but its `content` is blank/whitespace-only. | Same split as above. |
| `unresolved_citation_marker` | A raw, unresolved bracket marker (matching the same shape as `research_agent/report.py`'s `_UNRESOLVED_MARKER_RE`) leaked into rendered section content. | Direct analogue of R4's existing check. |
| `non_sequential_reference_numbering` | The `references` list's `number` values are not a clean `1..N` sequence. | Direct analogue of R4's existing check. |
| `orphan_reference` | A `references` entry whose `number` is never cited in any section's `reference_numbers`. | Direct analogue of R4's existing check. |
| `reference_source_unavailable` | A `references` entry's `paper_id`/`url` does not match any paper in `selected_papers` or any article in `approved_web_articles`. | **New in R6** — R4 has no equivalent check, because `report.py`'s own generation path is structurally incapable of producing this (the dynamic-Literal schema constrains citations to the exact offered set at generation time). R6 needs it because it evaluates *stored* report dicts, which can represent a state report.py's live path could never itself produce but a corrupted/regressed/hand-constructed one can — see fixture 8. |

R4's own two "warning, not hard" checks (`skipped_papers`) are **not**
promoted to hard failures here — R6 keeps them as an `informational_
signals`/`warnings` concern (§3), matching R4's own judgment that an
unused selected paper is expected, documented behavior, not
automatically a defect.

## 3. Informational-only deterministic signals

Documented now, **no thresholds attached** — attaching a threshold
without calibration evidence is exactly the "invented weights" R6A is
forbidden from doing (decision 2).

- **Section word counts** — per-section word count. Useful context for
  interpreting other signals (e.g. citation density) and for a future
  template-fit judge prompt, but **word-budget guidance in the
  generation prompts (`research_agent/report.py`'s per-section
  `description` strings) is not enforced anywhere in production today**
  — R6 must not invent an enforcement it doesn't actually have.
- **Citation density by section** — `[N]` marker count relative to
  section length. **High citation density is not automatically good**
  — a section can be densely cited and still misattribute every one of
  those citations (see fixture 4), or sparse and perfectly grounded.
- **Source citation frequency/share** — how many times each reference
  number is cited, across the whole report. **One dominant source is
  not automatically bad** — a report legitimately anchored on one
  central paper, with others in supporting roles, is a normal shape,
  not a defect on its own; this signal is context for a human or a
  future judge, not a rule.
- **Selected-source coverage** — how many of `selected_papers`/
  `approved_web_articles` were actually cited anywhere. **Every
  selected source need not appear** — this is the same
  `skipped_papers` signal R4 already treats as informational, carried
  forward as a rate rather than a raw list.
- **Skipped-paper rate** — `len(skipped_papers) / len(selected_papers)`,
  informational only, same reasoning as above.
- **Cross-section repetition proxy** (if later implemented) — e.g.
  n-gram overlap between sections, as a cheap deterministic *signal*
  toward the `coherence` judge dimension. Not implemented in R6A; noted
  here so R6B doesn't have to re-derive the idea from scratch.

## 4. Fixture architecture

```
eval_data/report_quality/
  README.md
  manifest.jsonl
  fixtures/
    *.json
```

**The manifest owns `id`, `path`, `tags`, `source_origin`, `notes`.**
Fixture bodies do not repeat `tags` — matching the instruction not to
duplicate tags into the fixture body absent an actual validation need
(there is none in R6A; R6B's loader reads tags from the manifest, the
same way the existing `_base.py::load_examples` reads `tags` from each
JSONL record today — the manifest is this suite's equivalent of that
per-record metadata, just split out because a report-quality fixture
is too large to comfortably live on one JSONL line, see
`eval_data/report_quality/README.md` for the full reasoning).

Fixture file shape (frozen for R6A):

```json
{
  "schema_version": "r6a-v1",
  "topic": "...",
  "template": "foundational|analytical|expert",
  "selected_papers": ["... Paper.to_dict()-shaped entries ..."],
  "approved_web_articles": ["... WebArticle.to_dict()-shaped entries ..."],
  "generated_report": {"... the real stored report-dict shape ..."},
  "expected": {
    "hard_failures": ["..."],
    "dimension_labels": {
      "citation_correctness": {"label": "pass|fail|not_applicable", "rationale": "..."},
      "groundedness": {"...": "..."},
      "synthesis_quality": {"...": "..."},
      "analytical_quality": {"...": "..."},
      "template_fit": {"...": "..."},
      "coherence": {"...": "..."},
      "source_balance": {"...": "..."}
    }
  },
  "human_annotations": [],
  "notes": "..."
}
```

**`expected.dimension_labels` is a synthetic, hand-constructed fixture
expectation — never human calibration data.** `human_annotations` is
the field reserved for real R6E annotations on real reports and is an
empty list on every R6A fixture, since none exist yet. No fixture in
this set, and no future R6B/R6C code, may treat the two as
interchangeable or describe `expected.dimension_labels` as "ground
truth" — it is a reviewable, hand-verifiable construction, and its own
rationale strings are what make it checkable, not an appeal to
authority.

`generated_report` uses the **real, current stored report-dict shape**
— confirmed against `research_agent/curation_session.py::
_serialize_report` and `research_agent/report.py`'s section-building
functions, not invented: one entry per section key (all 8 Analytical
keys — `executive_summary`, `introduction_scope`, `thematic_findings`,
`methodology_landscape`, `contradictions_open_debates`,
`gap_analysis`, `future_research_directions`, `conclusion` — plus the
three legacy projection keys `findings`/`limitations`/`future_scope`,
each a straight alias of its mapped Analytical section per
`_project_legacy_fields`), each shaped `{"content", "cited_papers",
"cited_web_articles", "reference_numbers"}`; a top-level
`skipped_papers` list; a top-level `references` list (`number`, `kind`,
`paper_id`/`url`, `title`, `formatted`, `link_url`); a top-level
`sections` list (`key`, `title`, `content`, `reference_numbers`, one
per Analytical key, in order); and `report_template`. Inline `[N]`
markers are written directly into each section's `content` string, at
their real rendered position — confirmed by reading
`_build_references_and_renumber` (`research_agent/report.py:743-875`),
which rewrites markers **in place**, never strips them from prose.
This is what makes sentence-level claim-to-citation checking possible
in R6C without inventing a new report representation.

## 5. Fixture evidence conventions

- All paper/web evidence in this fixture set is **synthetic and
  hand-written** — invented paper titles, authors, and abstracts;
  invented web articles. Nothing is copied from a real publication.
  `source_origin: "synthetic_handwritten"` on every R6A manifest entry
  records this explicitly.
- URLs use `example.com`/`papers.example.com`-style domains
  throughout — never a real arXiv ID, DOI, or live URL.
- No binary report exports (PDF/DOCX) are committed — fixtures store
  the report *dict*, matching how `research_agent/report.py` itself
  represents a report before export rendering; export correctness is
  already covered by `tests/test_report.py`'s own DOCX/PDF tests and is
  out of scope for a quality-evaluation fixture.
- Every fixture stays small enough to review directly in a PR diff —
  short synthetic abstracts/snippets and short report sections, not
  full-length prose at the real system's own target word budgets.

## 6. The 8 initial fixtures

See `eval_data/report_quality/README.md` for the per-fixture index
with tags. Three (`good_foundational`/`good_analytical`/`good_expert`)
deliberately share the exact same three synthetic papers and one web
article, so template-depth differences are the only variable between
them and a reviewer can compare all three against identical evidence.
The remaining five are adversarial/flawed by construction, each
isolating a different dimension or defect class (citation
misattribution, low-synthesis verbosity, source-side prompt injection,
report-prose evaluator-injection, and stacked structural corruption).

## 7. Known gap this fixture set surfaces (not fixed in R6A)

Fixture `source_prompt_injection` demonstrates a real, currently
undefended surface: `research_agent/qa.py`'s
`_detect_retrieved_prompt_injection` guard (added in R7E.5b) is wired
**only** into the chat/web-relevance filtering path
(`_filter_relevant_web_articles`, called from
`_filter_web_relevance_node` and `_accept_web_offer`). It is never
applied to paper abstracts anywhere in the system, and
`research_agent/report.py`'s own generation/evaluation/revision prompt
construction has no independent injection defense of its own — it
trusts whatever `selected_papers`/`web_articles` a session already
contains. A web article that reached a session via the R7B/R7E chat
path was checked once, at insertion time; a paper abstract pulled
directly from arXiv/Semantic Scholar search has never been checked by
anything, at any point.

**This is documented here as evaluation-surface context for the
fixture, not addressed by R6A.** R6A makes no change to
`research_agent/`. Closing this gap (e.g. applying an equivalent
injection guard inside `report.py`'s own prompt construction, or at
paper-ingestion time) is separate production work, to be tracked as
its own backlog item, not folded into an evaluation-only phase.

## 8. R6D — pairwise refinement evaluation (documented now, not built)

Not implemented in R6A. Frozen requirements for when R6D is actually
built:

- **Blinded A/B labels.** The judge sees `Report A`/`Report B`, never
  `draft`/`revised` — the draft/revised mapping is randomized per call
  and known only to the harness, not the judge.
- **Swapped order, both calls made.** Every pair is judged twice — once
  as `(draft=A, revised=B)`, once as `(draft=B, revised=A)` — never
  cached or short-circuited, since the entire point is detecting
  position bias.
- **Stable, seeded ordering.** Which physical report is labeled `A` on
  the *first* call is derived from a seed tied to the fixture/pair id,
  not re-randomized on every run — so a rerun of the same fixture is
  reproducible, and only the harness's own swap logic (not
  nondeterministic randomness) varies the second call's order.
- **A/B/tie per dimension**, not one global winner — a revision can
  legitimately win on `groundedness` and lose on `coherence` in the
  same pair; collapsing that into one winner would hide exactly the
  information a refinement-effectiveness measurement exists to surface.
- **`positional_disagreement`** reported as its own first-class metric
  — `True` whenever the two swapped calls disagree about which
  physical report won — never silently resolved by picking one call's
  answer over the other.
- **At least one human-labelled longer-but-not-better pair**, reserved
  for R6E — a pair where the revision is measurably longer than the
  draft but a human reviewer judged it no better (or worse), used
  specifically to catch a judge that defaults to preferring length.
  This fixture cannot be meaningfully synthetic/self-labelled the way
  the R6A single-report fixtures are, since the whole point is
  checking the judge against real human preference, not against a
  hand-authored expectation the same team also writes the judge
  prompt against.
- **Swap consistency is not proof of correctness.** A judge that
  agrees with itself across both orderings has passed a bias check,
  not a quality check — it can be perfectly self-consistent and still
  systematically wrong relative to a human reviewer. R6D's swap-order
  agreement rate and R6E's human-agreement rate are two different
  numbers, and neither substitutes for the other.

Citation-integrity preservation across a revision is checked
**deterministically**, not by the judge: `research_agent/report.py`'s
`revise_report` already calls `_restore_dropped_citations` to guarantee
a paper cited in the draft stays cited in the revision, but its own
docstring states web citations are **not** restored the same way (`revise_report`,
`research_agent/report.py:1844`) — R6D verifies both of these
documented behaviors hold, rather than assuming either.

## 9. R6B — future scoring semantics (frozen now)

- Deterministic hard-gate evaluators may return `score=1.0` or
  `score=0.0` and affect `passed`/`failed` in
  `research_agent/evals/runners/_base.py::run_suite`'s existing
  pass/fail convention (a fixture "passes" when every evaluator's score
  for it is exactly `1.0`) — this is the correct, narrow use of that
  convention, reserved for objectively checkable structural properties.
- Informational measurements (§3) return `score=None` (never counted
  against pass/fail, matching `run_suite`'s existing "a `None` score
  never fails the example" rule) or live entirely inside the
  prediction's own metadata, not as a scored evaluator result at all.
- **Future uncalibrated continuous judge scores must never be passed
  into `run_suite` as a pass/fail evaluator score.** A `0.62` from an
  R6C judge is not a `0.62` on the `1.0`-means-pass scale `run_suite`
  uses for deterministic checks — mixing the two scales would silently
  imply a calibration that doesn't exist yet.
- For a synthetic fixture with an explicit `expected.dimension_labels`
  categorical label, a **separate fixture-agreement evaluator** may
  score whether a *predicted* label matches the *expected* label
  (i.e., "did the harness's mocked/live judge output agree with the
  fixture's hand-written expectation" — a regression/plumbing check on
  the harness itself) — this is distinct from, and must never be
  conflated with, "is the judge actually right about report quality,"
  which only R6E's human-agreement measurement can speak to.
- **Unlabelled real reports are measured, not classified as
  passed/failed.** A real captured session's report has no
  `expected.dimension_labels` — R6B/R6C report its `judge_dimensions`
  and `informational_signals` as data, with no pass/fail verdict
  attached, since there is no fixture-side expectation to compare
  against.

## 10. R6C — future judge separation (frozen now)

- **Claim/source (citation-correctness + groundedness) judging is
  separate from holistic report judging** — two bounded calls per
  report, not one giant call (decision 4). The claim/source judge
  operates over extracted claim-source pairs (a claim sentence adjacent
  to its `[N]` marker, paired with reference `N`'s abstract/snippet
  text — see §4's confirmation that inline markers survive into final
  content, which is what makes this extraction possible at all).
- **Extracted claim-source pairs must be bounded/sampled
  transparently** — a report with many citations must not silently
  balloon into an unbounded prompt; whatever sampling/cap strategy R6C
  adopts must be visible in `judge_metadata`, not implicit.
- **The holistic judge covers `synthesis_quality`, `analytical_
  quality`, `template_fit`, `coherence`, and `source_balance`** — the
  five dimensions that need to see the whole report at once and can't
  be assessed from isolated claim/source pairs.
- **Judge model choice remains deferred to R6C itself**, not decided
  here — R6A freezes the *shape* of the judge output (§1), not which
  model produces it.
- **Same-model self-evaluation is not considered independent.** Using
  `research_agent/report.py`'s own `REPORT_MODEL` (`"gpt-4.1"`) as the
  R6C judge would repeat exactly the self-evaluation-bias problem R4
  already has (R4's generation, evaluation, and revision calls all
  default to that same constant, confirmed by reading `report.py`) —
  R6C's judge must be a genuinely different model.
- **No ensemble before human calibration.** Multiple judges/voting is
  not part of R6C's initial design — averaging together several
  judges of unknown individual accuracy doesn't produce a known
  accuracy; revisit only after R6E's human-agreement data exists.

## 11. Relationship to R4 (explicit, to prevent future drift)

R4's `evaluate_report` remains exactly what it is today: a pre-revision
gate inside report generation itself, bounded to one draft → evaluate
→ (at most one) revise → finalize flow, using the same model that wrote
the report. R6 does not replace, wrap, or re-score R4's own internal
decision — R6 is a separate, standing measurement system over
*already-produced* reports (fixture or real), built specifically
because R4's own architecture cannot answer "did the revision actually
help" (it never re-evaluates after revising — see
`refine_report_if_requested`'s own docstring, `research_agent/
report.py:1892-1964`, which documents `final_score` as `None`
post-revision by design) and was never independent of the model it's
grading in the first place.

## 12. R6C — what actually shipped (2026-08-10/11) — complete, R6C frozen

Sections 9-10 above froze the *design* R6C would follow. This section
records what was actually built against that design, and marks R6C
**closed** — see `specs/backend-backlog.md`'s R6C entry for the status
record and `docs/evaluation.md`'s "R6C.1" through "R6C.3" sections for
the full narrative.

**R6C.1 — bounded claim extraction, sampling, evidence registry,
injection sanitization** (`research_agent/evals/report_quality_inputs.py`):
- `build_evidence_registry` — deduplicated registry keyed by evidence
  id (`paper:<id>`/`web:<url>`), enriched from the report's own
  `references` list plus `selected_papers`/`approved_web_articles`.
  Runs the same independent injection detector R7E.5b's chat/web-relevance
  path uses against every source's text; a flagged source is marked
  `status="blocked_untrusted_source"` with `text=""` — its real content
  never reaches any judge.
- `extract_claim_units` — sentence-level claim extraction over the
  report's **raw** (unsanitized) content, split into `cited` (has a
  resolvable `[N]` marker, markers merged per sentence) and
  `uncited_candidate` (marker-free, ≥ a minimum word count) claim units.
  Deliberately raw, not sanitized — the claim/source judge's job is to
  fact-check what the report actually asserts, including a claim that
  is itself an injection attempt (see the security note below).
- `sample_claim_units` — bounded, transparent round-robin sampling
  across sections (`MAX_CITED_CLAIM_UNITS=16`, `MAX_UNCITED_CLAIM_
  CANDIDATES=8`), recorded in every live prediction's `judge_metadata.
  sampling_coverage` (per-section totals/selected counts, `truncated`
  flag) — satisfying decision 4/§10's "must be visible in
  `judge_metadata`, not implicit" requirement.
- `build_sanitized_report_and_findings` — a **separate** function,
  independent of claim extraction, that produces a redacted copy of
  each section's content (flagged sentences replaced with the literal
  `[BLOCKED_UNTRUSTED_INSTRUCTION]` placeholder) for the holistic
  judge only. The original report is never mutated.

**R6C.2 — two independent live judges**, wired into `predict_live`
(`research_agent/evals/runners/run_report_quality.py`):
- `research_agent/evals/judges/claim_source.py` — citation_correctness
  + groundedness, one bounded batched call, `CLAIM_SOURCE_JUDGE_
  PROMPT_VERSION` (now `"r6c2-claim-source-v3"`, see R6C.2c below).
- `research_agent/evals/judges/holistic.py` — synthesis_quality,
  analytical_quality, template_fit, coherence, source_balance, one
  call over the sanitized report, `HOLISTIC_JUDGE_PROMPT_VERSION =
  "r6c2-holistic-v1"` (unchanged since R6C.2).
- Both default to `REPORT_QUALITY_JUDGE_MODEL` (env-configurable,
  default `"gpt-5.6-terra"`), independent of `research_agent.report.
  REPORT_MODEL` ("gpt-4.1"), per decision "same-model self-evaluation
  is not considered independent" above. Structured outputs via a
  per-call `create_model` with `Literal`-constrained claim/evidence
  ids (the same technique `research_agent/qa.py`'s direct-relevance
  judge already proves live) — the model cannot invent or omit an id.
  No silent fallback: a malformed/refused response degrades the whole
  call to a recorded `error`, never a partially-trusted result.

**R6C.2a** added a fifth collective verdict, `not_a_verifiable_claim`,
to the claim/source judge — for uncited framing/organizational prose
that makes no externally verifiable research assertion (e.g. "Before
comparing these approaches, two terms are worth defining."). Rejected
as malformed if returned for a *cited* claim (a citation makes a
sentence a factual assertion by construction). Excluded entirely from
groundedness's judged set — neither pass nor fail nor unknown.

**R6C.2b/R6C.2c** — evidence-based fixture adjudication plus an
aggregation-policy recalibration, triggered by live findings (`report_
quality_history.csv` run_ids 2-5, commits `cf60191`, `bf8541d`,
`d0c4982`, `ff67113`): corrected demonstrably unsupported/under-cited
prose across the three `good_*` fixtures (CiteGuard metric
mischaracterization, an unsupported 3-way latency superlative,
cross-source under-citation, an unsupported "separate relevance
model" claim, an over-narrow "better passages" framing), then
recalibrated citation_correctness/groundedness aggregation from an
unversioned v1 rule ("any source verdict short of a clean `supports`
fails citation_correctness") to `r6c2-citation-aggregation-v2`
(`CITATION_AGGREGATION_POLICY_VERSION` in `run_report_quality.py`,
recorded in every live prediction's `judge_metadata`) — see §13 below
for the frozen semantics. `CLAIM_SOURCE_JUDGE_PROMPT_VERSION` bumped
v2→v3 with two literature-review-specific rules: bounded negative
claims ("none of the selected papers evaluates X" is checkable
against the supplied evidence set) and prospective recommendations
("future work should test X" does not assert X already happened).

**R6C.3** — a full 8-fixture live benchmark (run_id 6, commit
`2544e4e`) plus a bounded calibration audit that classified every
dimension mismatch as a fixture defect, a stale/incorrect expected
label, a judge-prompt/schema issue, an aggregation issue, an
intentional skip-semantics mismatch, or plausible model variability —
followed by one offline calibration pass (R6C.3a): 4 fixtures'
`expected.dimension_labels` corrected (`structural_and_metadata_
corruption` to `unknown` across all 7 dimensions, matching the
`predict_live` hard-failure-gate convention rather than the pre-R6C.1
`not_applicable` assumption; `citation_and_grounding_failure` and
`verbose_low_synthesis` template_fit/synthesis_quality corrected to
match the holistic rubric's own depth-inclusive definition;
`source_prompt_injection` recalibrated to `unknown`/`fail` as
appropriate), 6 fixtures' prose corrected for directly-evidenced
overreach (each edit verified against its cited abstract/snippet
before being applied — see `docs/evaluation.md`'s "R6C.3" section for
the sentence-level table), then 3 bounded targeted live reruns
(baseline `--tags baseline`, run_id 7; single-fixture stability check,
run_id 8; security `--tags security`, run_id 9) to confirm the
corrections held and no new material defect or injection bypass
appeared. **R6C is closed as of this checkpoint** — remaining
disagreement (documented in §14) is accepted, understood policy debt,
not silently unresolved.

## 13. Frozen aggregation semantics — `r6c2-citation-aggregation-v2`

`_aggregate_claim_source_dimensions` (`run_report_quality.py`) maps
the claim/source judge's per-claim verdicts onto its 2 owned
dimensions. This is a deliberately separate question from the R6A
`label` semantics in §1 — those are frozen; this is the *aggregation
rule* that produces one of those labels from many individual verdicts,
versioned independently (`CITATION_AGGREGATION_POLICY_VERSION`) so a
reader of a run's `judge_metadata` always knows which rule produced a
given `citation_correctness`/`groundedness` label.

**citation_correctness** — considers cited claims' per-source verdicts
only:
- `fail` if any attached source verdict is `does_not_support`.
- `unknown` if no source is `does_not_support` but at least one is
  `insufficient_evidence` (a real, inconclusive result — never
  silently folded into `not_applicable`).
- `pass` when every attached source is `supports` **or**
  `partially_supports` — both count as the source genuinely,
  relevantly contributing to the claim it was cited for. A source is
  not required to single-handedly cover a different attached source's
  own clause in a grouped comparative claim.
- `not_applicable` when no cited claim had any source verdict to
  judge at all.
- Persists `counts` (`supports`/`partially_supports`/`does_not_
  support`/`insufficient_evidence`) alongside the label.

**groundedness** — considers collective verdicts across cited and
uncited claims:
- `fail` if any collective verdict is `unsupported` or `partially_
  supported` — **this currently makes the whole report fail on a
  single such claim**; see §14 for why this is accepted, unresolved
  debt rather than a settled design.
- `unknown` if no collective verdict is a definite failure but at
  least one judged claim is `insufficient_evidence`.
- `pass` only when every judged claim's collective verdict is
  `supported`.
- `not_a_verifiable_claim` (R6C.2a) is excluded from the judged set
  entirely — neither pass, fail, nor unknown.
- `not_applicable` when nothing factual was ever judgeable.
- Persists matching `counts`.

**Continuous `score`/support-ratio values are diagnostic only.** Per
decision 2/§1, no calibration evidence exists to treat a `0.65` as
anything but informational context alongside the categorical `label`
— they are never averaged into an overall report-quality number and
never used to override the categorical pass/fail/unknown/not_
applicable determination above.

## 14. Accepted residual policy debt (not fixed, documented)

- **Groundedness's strict "any partial claim fails the report" rule
  does not cleanly separate all 8 synthetic fixtures.** Run_id 6's
  partial-claim rate ranged from 6.7% (`verbose_low_synthesis`) to
  64.7% (`good_analytical`) and 60.0% (the deliberately-bad `citation_
  and_grounding_failure`) — a "good" fixture and a deliberately-broken
  one can land at comparable or worse rates than each other under a
  binary label, even though the rates themselves clearly differ.
- **Analytical/Expert-template synthesis frequently introduces
  defensible, evidence-adjacent inference** (e.g. "ChunkRank assumes
  the correct passage is already in the top-k set" — a sound logical
  entailment from "reranks already-retrieved passages," never stated
  verbatim in any abstract) that the strict per-claim verifier marks
  `partially_supported`, indistinguishable in the current rule from a
  materially wrong claim.
- **Materiality/severity classification is a future improvement, not
  built.** No numeric threshold was invented from 8 synthetic
  fixtures — per decision 2, that would require real calibration
  evidence R6E is responsible for producing.
- **`good_foundational.template_fit` showed borderline stability**
  across 3 repeated live calls on the same-or-near-same content: pass
  (0.93, run_id 6) → fail (0.74, run_id 7) → pass (0.78, run_id 8),
  turning on how much explicit novice-level definition of terms like
  "single-hop"/"compressive memory" the judge considers sufficient.
  Documented as unresolved judge-calibration ambiguity, not "fixed" by
  any fixture edit.
- **Synthetic fixtures are not a substitute for a real, human-labelled
  dataset.** R6E's own real-report calibration work remains
  unaddressed by anything in R6C.
- **Judge cost/latency/reliability need broader, production-scale
  measurement.** Only 9 live runs exist as of this checkpoint (2-9
  targeted/full smoke runs); no throughput, rate-limit, or
  cost-at-scale data exists yet.
- **Evidence scope is abstracts/web snippets only**, never full papers
  — `judge_metadata.sampling_coverage.evidence_scope`'s own disclaimer
  states this explicitly on every live prediction.
- **`REPORT_QUALITY_JUDGE_MODEL` availability/pricing is environment/
  account dependent** — an invalid or inaccessible model id surfaces
  as a per-example judge failure (recorded, not a crash), not a
  guaranteed-available constant.
- **`.env` loading is not uniform across suite import paths.** Unlike
  `research_agent/api.py`/`research_agent/config/settings.py` and the
  `scripts/*.py` harnesses (each calling `load_dotenv()` directly),
  nothing under `research_agent/evals/` imports `research_agent.
  config.settings` or calls `load_dotenv()` itself — a live run
  depends on the shell already having credentials exported, or an
  explicit `uv run --env-file .env ...` invocation; this is now the
  documented command form for live report-quality runs.
