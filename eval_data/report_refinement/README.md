# Report refinement-effectiveness pair fixtures (R6D.1 + R6D.2 + R6D.3 + R6D.3a)

Fixtures for the `report_refinement` benchmark. **R6D.1** (schema/
fixtures/loader), **R6D.2** (deterministic/mock pair runner + CLI
registration), **R6D.3** (opt-in live semantic pair judging), and
**R6D.3a** (changed-claim + pairwise-holistic recalibration) are all
complete — see `docs/evaluation.md`'s "R6D.1"/"R6D.2"/"R6D.3"/"R6D.3a"
sections. There is still no call into `research_agent.report`'s
generation/evaluation/revision functions anywhere — R6D measures
whether R4's existing "refine once" step already changed a report's
structural and semantic state; it never performs or simulates a
refinement itself. **No claim is made here that refinement actually
improves report quality** — that requires real R4 output, which is
R6D.4's job (not started). R6D.3/R6D.3a only prove the live
*measurement* path works end-to-end, first against synthetic fixtures
with mocked judges, and now also calibrated against one real paid
pair's own evidence (run_id 3 — see `docs/evaluation.md`'s "R6D.3a"
section for the full story).

R6C is frozen at tag `r6c-report-quality-evaluation` and evaluates one
report independently across 7 dimensions (`citation_correctness`,
`groundedness`, `synthesis_quality`, `analytical_quality`,
`template_fit`, `coherence`, `source_balance`). R6D evaluates a
**pair** — an unrefined draft and a bounded refined report, same
topic/template/evidence — and measures **direction**
(`improved`/`unchanged`/`regressed`/`unknown`) per dimension, never an
overall score or winner. See `specs/report-quality-evaluation-plan.md`
section 8 for the original frozen R6D design and `docs/evaluation.md`'s
"R6D.1"/"R6D.2" sections for the full narrative.

## Running the suite

```bash
uv run python -m research_agent.evals.cli run --suite report_refinement --mode mock
uv run python -m research_agent.evals.cli run --suite report_refinement --mode mock --tags structural_integrity
uv run --env-file .env python -m research_agent.evals.cli run --suite report_refinement --mode live --subset 1
```

`--mode mock` (default) runs each pair's `draft_report` and
`refined_report` independently through R6B's own deterministic
hard-failure checks
(`research_agent.evals.runners.run_report_quality.predict()`, called
directly — never reimplemented) and derives a `hard_failure_direction`
(`improved`/`unchanged`/`regressed`/`mixed`) from the two resulting
failure sets. **The 7 semantic dimensions are never fabricated in mock
mode** — `dimension_directions` is always `null`, never inferred from
informational signals and never copied from a fixture's own
`expected.dimension_directions`. Zero OpenAI calls.

`--mode live` (R6D.3a, superseding R6D.3's own first design — see
`docs/evaluation.md`'s "R6D.3a" section for why) runs each side's
claim/source judgment independently (`run_report_quality.prepare_and_
judge_claims_only()`, reused directly, no standalone holistic call per
side), derives `citation_correctness`/`groundedness` direction from
ONLY the claim units that actually changed between draft and refined
(exact `claim_id` + field-equality matching — never fuzzy text
similarity, never an LLM), and derives the other 5 dimensions from a
SINGLE pairwise holistic call (`research_agent/evals/judges/
refinement_holistic.py`) that sees both reports together — **up to 3
real judge calls per pair** (one claim/source call per side, plus one
pairwise holistic call), dropping to 1 call when the identical-input
optimization applies for a `revision_applied=false` pair with
byte-identical reports (the pairwise holistic call is skipped entirely
in that case). Missing credentials exit 2 immediately, no CSV row, no
detail JSON, never a silent fallback to mock. See `docs/evaluation.md`'s
"R6D.3a" section for the full changed-claim comparison rules, the
pairwise holistic judge's schema, and the run_id 3 evidence that
motivated this recalibration. See `research_agent/evals/runners/run_
report_refinement.py`, `research_agent/evals/judges/refinement_
holistic.py`, and `research_agent/evals/evaluators/report_refinement.py`.

## Layout

```
eval_data/report_refinement/
  README.md          <- this file
  manifest.jsonl      <- one line per fixture: id, path, tags, source_origin, notes
  fixtures/*.json      <- the fixtures themselves
```

Same manifest+individual-files rationale as `eval_data/report_quality/`
— a pair fixture embeds TWO full report bodies plus shared evidence,
too large to comfortably fit one JSONL line.

## Pair schema (`schema_version: "r6d1-v1"`)

```json
{
  "schema_version": "r6d1-v1",
  "id": "...",
  "topic": "...",
  "template": "foundational|analytical|expert",
  "selected_papers": [ "... Paper.to_dict()-shaped entries, shared once at pair level ..." ],
  "approved_web_articles": [ "... shared once at pair level ..." ],
  "draft_report": { "... real stored report-dict shape ..." },
  "refined_report": { "... same shape ..." },
  "refinement_context": {
    "refinement_mode": "single",
    "revision_applied": true,
    "source_origin": "synthetic_handcrafted",
    "notes": "..."
  },
  "expected": {
    "hard_failure_direction": "improved|unchanged|regressed|unknown",
    "dimension_directions": {
      "citation_correctness": {"direction": "improved|unchanged|regressed|unknown", "rationale": "..."},
      "groundedness": {"...": "..."},
      "synthesis_quality": {"...": "..."},
      "analytical_quality": {"...": "..."},
      "template_fit": {"...": "..."},
      "coherence": {"...": "..."},
      "source_balance": {"...": "..."}
    }
  }
}
```

**No `overall_direction`, `overall_score`, `accept_refinement`, or
`winner` field exists anywhere in this schema, on purpose.** Those are
calibration decisions for a later phase (per `specs/
report-quality-evaluation-plan.md` section 8's own "no invented
weights" posture) — `research_agent/evals/report_refinement_inputs.py`'s
loader actively rejects any fixture that includes one.

`draft_report`/`refined_report` reuse the exact real stored report-
dict shape R6A/R6C already validated (`report_template` + the 8
canonical Analytical section keys, each `{"content",
"reference_numbers"}` + `references` + `sections`) — confirmed against
`research_agent/curation_session.py::_serialize_report` — so R6C's own
preparation code can be pointed at either half of a pair later without
reshaping anything. Two deliberate simplifications versus R6A/R6C
fixtures, both because a *pair* needs two full report bodies where a
single-report fixture needs one: the 3 legacy projection keys
(`findings`/`limitations`/`future_scope`) are omitted (pure aliases,
never read by any deterministic check or claim extraction), and
per-section `cited_papers`/`cited_web_articles` are omitted (also
never read by anything — confirmed by grep against
`research_agent/evals/report_quality_inputs.py` and `runners/
run_report_quality.py` during R6C).

## Pair invariants (enforced by the loader, not just documented)

1. `id` is unique and matches the manifest.
2. `schema_version` is exactly `r6d1-v1`.
3. Manifest path resolves strictly inside `eval_data/report_refinement/`
   (a `../` traversal is rejected).
4. `draft_report` and `refined_report` use the same `report_template`.
5. The pair's own `template` matches both reports.
6. `selected_papers`/`approved_web_articles` are shared once at the
   pair level — a report that embeds its own copy of either key is
   rejected outright, so there is no way for per-report evidence to
   silently diverge.
7. Both reports use the existing 8 canonical section keys, in
   canonical order (`sections` list order is checked directly, and
   mirrored against each section's own `content`).
8. References/inline markers are structurally valid (the same 6 R6A/
   R6B hard-failure identifiers, via an independent, R6D-owned
   `check_structural_validity` — never imports R6C's own checks) on
   both reports, *unless* `expected.hard_failure_direction` declares
   `improved`/`regressed`, in which case the loader requires the
   correct side to actually be the clean one and the other to actually
   have at least one detected hard failure.
9. Every paper/web source either report cites resolves inside the
   shared evidence pool — this is `check_structural_validity`'s own
   `reference_source_unavailable` check, reused rather than duplicated.
10. Every one of the 7 R6C dimensions has exactly one expected
    direction (one of `improved`/`unchanged`/`regressed`/`unknown`)
    and a non-empty rationale — no missing, no extra dimension.
11. `unknown` is reserved for a genuinely unjudgeable comparison (most
    commonly: a structural hard failure gates off live judgment on one
    side entirely, per R6C's own convention). The loader cannot
    mechanically verify *why* a fixture author chose `unknown` — this
    is a documented authoring discipline, not a field the code can
    police beyond requiring it be one of the 4 valid values.
12. `revision_applied=false` requires `draft_report == refined_report`
    exactly (deep equality) — enforced, not just documented.
13. `revision_applied=true` requires *some* report-body difference —
    a fixture claiming revision happened with byte-identical reports
    is rejected.
14. No `expected.dimension_directions[...].rationale` string may
    appear verbatim inside either report's own section content — a
    fixture's answer key must never leak into the judge-ready report
    text a future judge would actually read.

## The 7 fixtures

| id | template | tags | intended direction |
|---|---|---|---|
| `clear_grounding_improvement` | foundational | `groundedness`, `improvement` | Draft overclaims "eliminates all" unsupported claims; refined restates the same finding accurately with the same citation. **groundedness improved, citation_correctness improved** (R6D.3a: the one changed claim's attached source flips `does_not_support` → `supports`), everything else unchanged. |
| `holistic_synthesis_improvement` | analytical | `synthesis_quality`, `analytical_quality`, `coherence`, `improvement` | Draft is a paper-by-paper listing with a near-duplicate Gap Analysis/Future Research Directions pair; refined adds genuine cross-source comparison (grouped `[1][2]` citation) and a distinct, concrete proposal. **synthesis_quality, analytical_quality, coherence improved**; citation/groundedness unchanged. |
| `justified_no_revision` | expert | `tie`, `no_revision` | `revision_applied=false`; `draft_report`/`refined_report` are byte-identical. **All 7 directions unchanged.** |
| `cosmetic_rewrite_tie` | foundational | `tie`, `cosmetic` | `revision_applied=true`; every section is reworded, no claim/citation/structure changes. **All 7 directions unchanged** — demonstrates that different prose is not automatically improvement. |
| `citation_regression` | analytical | `citation_correctness`, `groundedness`, `regression` | Refined prose reads more smoothly but silently misattributes one paper's finding to the other paper's reference number. **citation_correctness and groundedness regressed**, the rest unchanged. |
| `mixed_tradeoff` | expert | `synthesis_quality`, `coherence`, `groundedness`, `tradeoff` | Refined genuinely improves cross-source synthesis (paper + web evidence combined into one narrative) but adds an unsupported "now the industry-standard best practice" overclaim. **synthesis_quality and coherence improved, groundedness regressed** — a real tradeoff, with no winner field to collapse it into one number. |
| `structural_regression` | analytical | `structural_integrity`, `hard_failure`, `regression` | Refined strips every inline citation marker, orphaning the only reference. **`hard_failure_direction=regressed`**; all 7 judge-dependent dimensions **`unknown`** (structural gating prevents a fair comparison on the broken side, matching R6C's own hard-failure-gate convention). |

Between them, the 7 fixtures cover all three templates (foundational
×2, analytical ×3, expert ×2), both paper-only and paper+web evidence
(`mixed_tradeoff`), and a grouped `[1][2]` citation
(`holistic_synthesis_improvement`). No injection-attack fixtures exist
here — R6C's own `source_prompt_injection`/`evaluator_injection_in_
report` fixtures already validate judge-input security; R6D.1 does not
duplicate that.

## Fixture evidence conventions

Two fresh synthetic sources, distinct from R6A/R6C's ChunkRank/
LongMem/CiteGuard set (to avoid any risk of the two benchmarks being
confused with each other): **SpanCite** (`paper_id: "spancite-2024"`,
a decoding-time sentence-to-source-span traceability constraint) and
**DriftGuard** (`paper_id: "driftguard-2024"`, a multi-turn topic-
drift passage filter), plus one web article on production citation
grounding. All evidence is synthetic and hand-written —
`source_origin: "synthetic_handcrafted"` on every manifest entry and
every fixture's own `refinement_context.source_origin`, matching
R6D.1's frozen schema. `example.com`/`papers.example.com` domains
throughout, same convention as `eval_data/report_quality/`.

## Loader

`research_agent/evals/report_refinement_inputs.py` — manifest+fixture
loading, all 14 invariants above, `check_structural_validity` (the 6
R6A/R6B hard-failure identifiers, independently implemented),
`reports_are_equal`/`diff_report_sections` (deterministic content
equality/diff helpers for R6D.2), and the canonical `REQUIRED_
DIMENSION_NAMES`/`VALID_DIRECTIONS`/`VALID_TEMPLATES` constants so
R6D.2 never has to redefine them. Raises `ReportRefinementFixtureError`
on any invariant violation. Zero API calls, no import of `openai`,
`research_agent.report`, or R6C's own runner/preparation modules (see
`tests/test_evals_report_refinement.py::TestNoOpenAIOrNetworkPath`).

```python
from research_agent.evals import report_refinement_inputs as rri

examples = rri.load_report_refinement_examples()          # all 7
examples = rri.load_report_refinement_examples(tags=["improvement"])
examples = rri.load_report_refinement_examples(subset=2)
```

## What R6D.4 is (not built here)

R6D.1-R6D.3 evaluate only these 7 synthetic, hand-authored pair
fixtures. R6D.4 (not started) is responsible for running this same
live evaluation path against *real* R4-generated draft/refined report
pairs, to actually answer whether refinement improves report quality
in practice — see `specs/backend-backlog.md`'s R6D entry.
