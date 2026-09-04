# Public Multi-User Deployment — Architecture Review (Part 2)

**Status:** read-only review document. Nothing in this file has been implemented,
provisioned, deployed, or pushed. No production behavior was changed to produce
it.

This is the continuation of a two-part review. Part 1 (sections A–H) was
delivered in conversation and is not reproduced here in full — only a
condensed recap, the six corrections requested against it, and the
previously-missing sections (I–M) follow.

## Condensed recap of sections A–H

- **A.** HEAD `21bd8fc7eb66fd1dd2449ee3b68113b6960147d9`, `main` ahead of
  `origin/main` by 5 (unpushed H5 prompt-injection commits, confirmed present),
  clean tree. No user-identity concept anywhere in the code today; Basic Auth
  is one shared static credential. No ownership checks on any route. Stores:
  `history.sqlite` (452 KB), `qa_checkpoints.sqlite` (772 MB, dev/e2e debris),
  `usage_telemetry.sqlite` (512 KB), `chroma_db/` (167 MB), `cache/` (187 MB).
- **B.** Corrected goal: genuine multi-tenant public product — self-service
  sign-up, per-user data isolation enforced at the data layer, two-sided cost
  governance (per-user + global), a mentor who can register like any stranger.
- **C.** The prior allowlisted-IAP/SQLite plan was rejected: not self-service,
  no application-level user model, a global admission budget sized for a demo
  rather than strangers, and no abuse-surface work at all.
- **D.** Auth comparison across 7 options; recommendation was Firebase
  Authentication — refined further below (Correction 1).
- **E.** Move `users` / `saved_searches.owner_id` / new `curation_owners` to
  Postgres now; keep LangGraph `SqliteSaver`, telemetry, admission, leases,
  and caches on local SQLite for the beta; Chroma stays global/shared on the
  VM disk; PostgresSaver migration investigated and deferred.
- **F.** Ownership schema (`users`, `owner_id`, opaque `thread_id`s, 404-not-403
  cross-owner) and a full endpoint permission matrix.
- **G.** Usage/abuse design reusing `admission.py`/`leases.py`/`telemetry.py`:
  per-user + per-session budgets, a tunable global ceiling, an admin kill
  switch, IP throttling at the edge, an approval gate for new accounts.
- **H.** GCE VM + Docker + Cloud SQL recommended over VM+local-Postgres,
  Cloud Run+Cloud SQL+external vector store, and VM+SQLite-only; sizing
  `e2-standard-2` (8 GiB) to start, refined further below (Correction 6).

---

## Corrections applied in this revision

### Correction 1 — Firebase Authentication vs. Google Identity Platform

They are **the same underlying service**. "Identity Platform" is Firebase
Authentication exposed through the Google Cloud console with enterprise
features layered on top (multi-tenancy, SAML/OIDC federation, MFA at a paid
tier) and a different, usage-based billing model once enabled. Firebase
Authentication is the same backend, accessed via the Firebase console/SDKs,
free for standard providers (Google, email/password, etc.) at generous volume.

**Exact recommendation:** use **Firebase Authentication**, not Identity
Platform, with **only the Google provider enabled**. Link the existing GCP
project into Firebase (do not create a second, disconnected project) so IAM
and billing stay unified. If multi-tenancy or SAML is ever genuinely needed
(Tier 3, unrestricted production, enterprise customers), upgrading the same
project into Identity Platform mode later is a **console configuration
change**, not a data migration — user records and `uid`s carry over
unchanged. Do not provision Identity Platform now; it adds cost/config
surface this beta does not need.

### Correction 2 — no service-account JSON key; attached SA + ADC

**No Firebase Admin service-account key file is created, downloaded, stored
in Secret Manager, or baked into any image, at any tier of this plan.**

- **Token verification** (the hot path, every request): does **not** require
  any credential at all. A Firebase ID token is a JWT signed by Google;
  verifying it means checking the signature against Google's public JWKS
  (fetched over plain HTTPS) plus `aud`/`iss`/`exp`. This can be done with
  `google-auth`'s `google.oauth2.id_token.verify_firebase_token(...)`, or
  `firebase_admin.auth.verify_id_token(...)` initialized with **no explicit
  credential** — either way, no key file, no secret.
- **Admin operations** (account deletion via the Firebase user store, setting
  custom claims): these do call an authenticated Google API and need a
  credential — but that credential should be the **VM's attached service
  account resolved via Application Default Credentials (ADC)**, exactly the
  same mechanism `google.auth.default()` and `firebase_admin.initialize_app()`
  (with no credential argument) use automatically on GCE. No key file is
  involved; the GCE metadata server supplies short-lived tokens transparently.
- **Required IAM roles on the VM's attached (dedicated, non-default) service
  account** — least privilege, granted at the project or resource level as
  noted:
  | Role | Purpose |
  |---|---|
  | `roles/firebaseauth.admin` | Admin SDK operations (account deletion, custom claims) via ADC — omit if account deletion is implemented purely as a Postgres-side soft delete for the 7-day beta (see Day 8–9 below) and add it only when real Firebase-user-record deletion is built |
  | `roles/cloudsql.client` | Connect to Cloud SQL via the Cloud SQL Auth Proxy using ADC, no downloaded key |
  | `roles/cloudsql.instanceUser` (if IAM DB auth is used instead of a static password) | Maps the service account to a Postgres IAM user — preferred over a Secret-Manager password where practical |
  | `roles/secretmanager.secretAccessor` | Read any remaining secrets (provider API keys, and a DB password only if IAM DB auth is not used) at container start |
  | `roles/logging.logWriter`, `roles/monitoring.metricWriter` | Standard observability; explicit here because a dedicated SA does not inherit the default SA's broad grants |
  Do **not** grant `roles/editor`/`roles/owner`, and do not reuse the
  project's default Compute Engine service account (it is typically
  over-scoped) — create a dedicated SA for this VM.

### Correction 3 — three explicitly separated tiers

| Tier | Definition | Scope of this document |
|---|---|---|
| **Tier 1 — Seven-day approval-gated controlled beta** | Real Firebase auth, real per-user ownership, real per-user + global budgets, one VM + Cloud SQL, **new accounts default `approved = false`**, a tester cap, fresh production data. Anyone can *sign up*; using the product requires an operator's approval. Not self-service in the full sense — deliberately so. | Section I |
| **Tier 2 — Self-service public beta** | The approval gate is raised or removed (auto-approve, or approval only for abuse signals); IP throttling / Cloud Armor / signup-abuse protection become load-bearing (not optional, since anyone can now both register *and* use it immediately); full account-deletion lifecycle; admin visibility; connection pool tuned under real load. Still one VM, still `SqliteSaver`, still not horizontally scaled. | Section J |
| **Tier 3 — Unrestricted/scalable production** | Horizontal app scaling, which forces `PostgresSaver` (Correction 5) and a Chroma migration off single-VM `PersistentClient`; CI/CD; admin dashboard; billing; SLAs. **Not attempted by either schedule below.** Triggered only by a concrete, evidenced decision to run more than one app instance. | Out of scope; named only in Part L |

Every "controlled beta" claim in Section I and every "public beta" claim in
Section J must be read against this table — neither schedule produces Tier 3,
and Section I explicitly does not produce Tier 2 either.

### Correction 4 — Cloud SQL `curation_owners` vs. SQLite LangGraph checkpoints: consistency model

Two independent engines, no distributed transaction. **The Postgres
`curation_owners` row is the single reachability source of truth; the SQLite
checkpoint is secondary.** This gives one simple, fail-closed rule instead of
two-phase-commit machinery neither engine supports together.

- **Creation ordering** (`POST /curation/start`): write the `curation_owners`
  row **first** (Postgres), then invoke the LangGraph graph to create the
  initial checkpoint (SQLite).
  - Owner-row write fails → return an error to the client; the checkpointer
    is never invoked; nothing is orphaned.
  - Owner-row write succeeds, checkpoint write fails → the owner row exists
    with no matching checkpoint. **Fail closed**: any subsequent per-resource
    read (`GET /curation/{id}`) for that `session_id` treats "owner row
    present, `load_curation_session` returns `None`" as **404**, never a
    500 or a partially-constructed session. The client already received an
    error from the original `/curation/start` call and is expected to retry
    (which mints a *new* `session_id`); the stranded owner row is swept by
    reconciliation (below), not repaired in place.
- **Deletion ordering** (`DELETE /curation/{id}`): delete the
  `curation_owners` row **first** (Postgres), then delete the checkpoint
  thread (SQLite, `delete_curation_session`).
  - This is the fail-closed choice for deletion specifically: the instant
    the owner row is gone, the session is unreachable and invisible to the
    user — correct "deleted" behavior — even if the checkpoint bytes linger
    briefly. The reverse order (checkpoint-first) risks a live owner row
    still listing a session whose backing state is already gone, which is a
    broken-UX case, not a security case, but is strictly worse than
    "briefly orphaned bytes nobody can reach."
- **Orphan classes and handling**:
  1. *Owner row, no checkpoint* (failed create, or checkpoint deleted out of
     band) — never exposed as anything but 404 on read; a reconciliation job
     deletes owner rows older than a safety window (e.g. 1 hour) with no
     matching checkpoint.
  2. *Checkpoint, no owner row* (failed delete ordering edge case, or a bug)
     — never exposed to any user (`list`/`get` only ever read from
     `curation_owners`); a reconciliation job periodically diffs checkpoint
     `thread_id`s against `curation_owners.session_id` and calls
     `delete_thread()` on anything unowned older than a longer safety window
     (e.g. 24 hours, to avoid racing an in-flight create).
  - The reconciliation job **never invents an owner for an orphaned
    checkpoint** — assigning abandoned data to whoever runs the job is a
    security bug, not a repair.
- **Tests required** (new, before Day 4's ownership-enforcement work is
  considered complete):
  1. Mocked Postgres-write-failure during `/curation/start` → assert the
     checkpointer is never invoked.
  2. Mocked checkpoint-write-failure during `/curation/start` (owner row
     succeeds) → assert `GET /curation/{id}` returns 404, not 500.
  3. Normal delete → assert both the owner row and the checkpoint are gone.
  4. Mocked checkpoint-delete-failure during `DELETE /curation/{id}` (owner
     row already deleted) → assert the session is invisible via `list`/`get`
     (404) despite lingering checkpoint bytes; assert the reconciliation job
     later removes them.
  5. Reconciliation-job unit tests: seed both orphan classes, run the job,
     assert exactly the intended rows/threads are removed and nothing live
     is touched.
  6. Two near-simultaneous `/curation/start` calls for the same user do not
     corrupt either store (each call mints an independent `session_id` —
     assert explicitly, not just assumed).

### Correction 5 — exactly when `SqliteSaver` becomes unacceptable

`SqliteSaver` is a single-file, single-process, single-writer store. It
becomes **unacceptable, not merely suboptimal**, the moment any one of these
becomes true — and `PostgresSaver` becomes a **mandatory prerequisite**, not
an optional upgrade, at that point:

1. **Horizontal scaling** — more than one app process/instance (multiple
   uvicorn workers, multiple VMs, or a managed autoscaler) needs to read or
   write the same checkpoint data. Two writers against one SQLite file is not
   a supported configuration.
2. **Zero-downtime deployment** — a rolling deploy briefly runs old and new
   versions against the same store simultaneously; this is the ">1 process"
   case above under a different name.
3. **Ephemeral/decoupled compute** — moving to Cloud Run or an autoscaling
   managed instance group without a single shared, always-attached disk.
   SQLite's file must live on the same machine as the one process that owns
   it; anything that decouples compute from a fixed disk makes local SQLite
   **structurally impossible**, not just risky.
4. **Measured write contention or file-growth pressure** — `SQLITE_BUSY` /
   lock-wait errors appearing in logs under real load, or the checkpoint
   file's growth rate (each turn persists the *entire* session dict — this
   is exactly why the dev file reached 772 MB) degrading I/O latency.

**None of these triggers are expected to fire in Tier 1 or Tier 2** — both
are explicitly one VM, one process, no autoscaling, and both accept a brief
restart window as a documented cost of staying single-instance. Watch
`SQLITE_BUSY` frequency and checkpoint-file growth rate as the two leading
indicators; treat either crossing a defined threshold as the trigger to
schedule the `PostgresSaver` migration spike — not user count or
public-vs-private status by themselves. `PostgresSaver` only becomes
mandatory when a concrete decision is made to run more than one app
instance (Tier 3); it should not be pre-built speculatively.

### Correction 6 — all costs provisional

**Every dollar figure anywhere in this review (this document and the prior
conversation turn) is a provisional order-of-magnitude estimate from public
list pricing at the time of writing.** None of it is a quote. Before
committing any budget or writing it into a plan as a constraint, verify the
actual figures with the **GCP Pricing Calculator**
(https://cloud.google.com/products/calculator) for the specific selected
region, machine types, Cloud SQL tier, and expected egress — regional
pricing, sustained-use discounts, and list-price changes since this review
was written can all move the real number materially. Every schedule item
below that references a machine/Cloud SQL tier is a **sizing**
recommendation, not a cost commitment.

---

## I. Complete seven-day controlled-beta schedule (Tier 1)

Scope per Correction 3: Firebase auth (Google only), Postgres ownership,
SQLite checkpoints, one VM + Cloud SQL, **approval-gated** — a mentor can
register but needs approval before using paid features; not self-service in
the full sense; not Tier 2 or Tier 3.

### Day 1 — Freeze, safety rails, identity decision
- **Objective:** freeze scope to this document; push the 5 local H5 commits;
  stand up one Firebase project (linked to the target GCP project) with only
  the Google provider enabled; fix the two cheap concurrency risks already
  identified (`/health` on the threadpool, unbounded AnyIO ceiling).
- **Files/modules:** `research_agent/api_app/routers/health.py` (→
  `async def`), `research_agent/api_app/app.py` (`lifespan()`: set the AnyIO
  thread limiter explicitly), `docs/plans/public-multi-user-deployment-review.md`
  (this file, committed as the frozen scope reference).
- **Tests:** full existing `uv run pytest` + `npm test` + `npm run build`
  green, unchanged. A manual check: a synthetic 60s-blocking sync request
  does not delay a concurrent `/health` call.
- **Stopping condition:** H5 pushed; Firebase project exists with Google
  provider enabled and no other provider; `/health` is `async def`; the
  limiter is set and logged at startup.
- **Completion evidence:** `git log`/`git status` showing `main` in sync with
  `origin/main`; a screenshot or console listing of the Firebase project's
  enabled providers (Google only); a passing test run transcript; the manual
  `/health`-not-starved check's timing output.

### Day 2 — Postgres ownership schema + creation/deletion consistency (Correction 4)
- **Objective:** create `users`, `curation_owners`, `searches.owner_id` in
  Postgres (local/dev instance for now — no GCP provisioning yet); implement
  the create-first/delete-first ordering and the fail-closed 404 rule from
  Correction 4; write the reconciliation job as a standalone script (not yet
  scheduled).
- **Files/modules:** new `research_agent/db/` (or similar) module for the
  Postgres connection + schema, `research_agent/storage.py` (additive
  `owner_id` column, following the existing `ALTER TABLE ADD COLUMN`
  pattern), a new service-layer wrapper around `save_curation_session`/
  `delete_curation_session` implementing the two-step ordering, a new
  `scripts/reconcile_curation_ownership.py`.
- **Tests:** the six Correction-4 test cases in full (mocked-failure creation
  and deletion ordering, orphan classes, reconciliation-job unit tests,
  concurrent-create independence) plus existing migration/back-compat tests
  (pre-existing `searches` rows with `owner_id IS NULL` still load and are
  invisible to every authenticated user).
- **Stopping condition:** all six Correction-4 tests pass; no owner-check
  enforcement is wired into routes yet (that is Day 4); existing suite green.
- **Completion evidence:** test run output naming all six new test cases by
  name; a manual run of `scripts/reconcile_curation_ownership.py` against a
  seeded orphan of each class, showing exactly the intended rows removed.

### Day 3 — Identity middleware + token verification
- **Objective:** verify Firebase ID tokens with no service-account key
  (Correction 2): implement `IdentityMiddleware` (Bearer header, not a
  cookie) using either `google-auth`'s JWKS verification or
  `firebase_admin.auth.verify_id_token` with no explicit credential; add
  `get_current_user`; add `GET /me`; require identity on every route except
  `GET /health`.
- **Files/modules:** new `research_agent/identity_middleware.py`,
  `research_agent/api_app/app.py` (register outermost, same slot as the
  current `BasicAuthMiddleware`, which is retained but disabled via
  `AUTH_ENABLED=false` rather than deleted), a new router for `GET /me`, a
  test-suite autouse fixture injecting a synthetic verified identity.
- **Tests:** token verification unit tests — valid, expired, malformed,
  wrong-audience, wrong-issuer, missing header, duplicate header; the full
  existing suite green under the synthetic-identity fixture.
- **Stopping condition:** every non-`/health` route returns 401 with no
  token; a valid token reaches the route; the full suite is green.
- **Completion evidence:** the token-verification test file's pass output;
  a manual `curl` showing 401 with no token and 200 with a real signed test
  token (Firebase Auth emulator or a real test account token, no paid call
  involved).

### Day 4 — Ownership enforcement across every route
- **Objective:** wire the Part F permission matrix into every listed route:
  owner lookup before checkpointer/Chroma access, **404** (never 403) on
  mismatch or on the Correction-4 "owner row exists, checkpoint doesn't"
  case; `GET /curation/reviews` and `GET /library` filtered by caller.
- **Files/modules:** every curation/library router (`api_app/routers/
  curation_*.py`, `library.py`) and their backing services.
- **Tests:** a new full permission-matrix test file — owner→200, a different
  authenticated user→404, unauthenticated→401 — for every route in the Part
  F matrix, including the streaming routes (ownership checked **before**
  any `StreamingResponse` is constructed, with a provider-call tripwire
  proving no provider call happens on a rejected stream).
- **Stopping condition:** the full permission matrix is green, including the
  streaming-ownership-before-provider-call tripwire test.
- **Completion evidence:** the permission-matrix test file's pass output,
  itemized by route; the tripwire test's assertion output showing zero
  provider calls on a rejected request.

### Day 5 — Frontend auth states
- **Objective:** Firebase JS SDK, a Google sign-in screen, token attachment
  in the shared fetch wrapper, 401 handling (redirect to sign-in), sign-out,
  and gating `CurationWorkspacePage` on an authenticated state.
- **Files/modules:** `frontend/src/App.tsx`, `frontend/src/lib/api/client.ts`
  (attach `Authorization: Bearer <token>` in `request()`), a new
  `frontend/src/components/Auth/` directory, `frontend/src/pages/
  CurationWorkspacePage.tsx` (render nothing/a sign-in prompt when logged
  out).
- **Tests:** new frontend auth-state tests (logged-out render, token
  attachment, 401→redirect, sign-out clears state); existing frontend suite
  green; production build clean.
- **Stopping condition:** two distinct Firebase test accounts, exercised via
  the dev shim or the Firebase Auth emulator, see disjoint `GET /curation/
  reviews` results in a manual check; frontend suite and build both green.
- **Completion evidence:** the new auth-state test file's pass output; a
  manual two-account screenshot/log showing disjoint review lists;
  `npm run build` output.

### Day 6 — Infrastructure provisioning
- **Objective:** provision the VM (dedicated non-default service account per
  Correction 2, with exactly the listed IAM roles), Cloud SQL instance,
  persistent disk + swap, HTTPS (Caddy/nginx + Let's Encrypt on the VM,
  since IAP is no longer the auth mechanism — no external load balancer
  required for this tier), Secret Manager for any remaining secrets (no
  Firebase key file, per Correction 2), Artifact Registry for the image,
  firewall rules (443 public only; SSH via IAP-for-TCP or OS Login; Cloud
  SQL private-IP/proxy only), deploy by image digest, wire the **approval
  gate** (`users.approved DEFAULT false`; budgets return 403 until an
  operator flips it), tune the global admission ceiling for real deployment,
  run one backup drill (Cloud SQL automated backup + `scripts/
  data_backup.py` for the VM's `/app/data`) and one rollback drill (deploy a
  dummy image, roll back to the prior digest).
- **Files/modules:** infra-only — Dockerfile unchanged, a new deployment
  runbook doc, systemd unit or `docker-compose.yml` referencing the image
  digest.
- **Tests:** none new in-repo; operational verification only.
- **Stopping condition:** a real Google account can sign up over HTTPS and
  is denied real usage until approved; an approved account works end-to-end;
  `docker history`/image inspection shows no baked secret and no Firebase
  key file; the backup drill and the rollback drill both complete
  successfully.
- **Completion evidence:** the deployment runbook committed; backup-drill
  and rollback-drill transcripts (timestamps, commands, before/after
  `/health` status); a note recording the actual Cloud SQL tier and VM size
  chosen, with a placeholder for the Pricing-Calculator-verified monthly
  estimate (Correction 6 — not filled with an unverified number).

### Day 7 — Bounded multi-user journey, measurement, and reporting
- **Objective:** execute the full acceptance matrix (Section K); run a real
  two-user journey (Pratik + mentor, mentor self-registers and is approved
  live); run a small synthetic load test within the approved-tester cap;
  produce the residual-risk report explicitly labeled as **Tier 1 —
  approval-gated controlled beta**, not a public or unrestricted claim.
- **Files/modules:** none — verification and documentation only.
- **Tests:** the full Section K acceptance matrix.
- **Stopping condition:** every Section K test passes or is explicitly
  waived with a written rationale; peak RSS/CPU/latency/error-rate/`/health`
  p95/disk-growth-rate are recorded from the load test; the VM-sizing
  decision (stay at the provisioned size vs. resize) is recorded with
  supporting numbers.
- **Completion evidence:** the acceptance-matrix results (pass/waived per
  item); the load-test metrics table; the residual-risk report file, with
  its title/summary explicitly stating "approval-gated controlled beta."

---

## J. Complete 10–14-day public-beta schedule (Tier 2)

Builds on Days 1–7. Scope per Correction 3: the approval gate is raised/
removed, so IP throttling and signup-abuse protection become load-bearing;
account deletion becomes a real, tested lifecycle, not a documented
manual process; admin visibility exists; the connection pool is tuned under
real concurrent load. Still one VM, still `SqliteSaver` (Correction 5's
triggers are not expected to fire here) — **this tier is still not Tier 3.**

### Day 8 — Account-deletion lifecycle, part 1: soft delete
- **Objective:** implement `users.deleted_at` soft delete, immediate
  loss of session validity, and the partial-unique-constraint exception
  that frees the email for reuse only after soft delete.
- **Files/modules:** the `users` schema/migration, the identity-verification
  path (a soft-deleted user's token must be rejected even if still
  cryptographically valid), a new `/account` deletion endpoint.
- **Tests:** a soft-deleted user's subsequent requests return 401; the freed
  email can be used by a new sign-up; the old `user_id` is never reused.
- **Stopping condition:** the three tests above pass; existing suite green.
- **Completion evidence:** test output; a manual soft-delete-then-re-signup
  demonstration with two distinct emails.

### Day 9 — Account-deletion lifecycle, part 2: async purge job
- **Objective:** a bounded async job that, within a documented SLA (e.g. 30
  days), deletes the user's `curation_owners` rows (via the existing
  `delete_curation_session`, following Correction 4's deletion ordering) and
  `searches` rows; Chroma's shared paper content is never touched.
- **Files/modules:** a new scheduled script alongside
  `scripts/reconcile_curation_ownership.py`.
- **Tests:** a seeded soft-deleted user with real owned rows, run the purge
  job, assert exactly their rows are gone and nothing else is touched;
  assert Chroma content is unaffected.
- **Stopping condition:** the purge test passes; the SLA is documented in
  the deployment runbook.
- **Completion evidence:** test output; the runbook entry stating the SLA.

### Day 10 — IP-level throttling and Cloud Armor
- **Objective:** rate-based Cloud Armor rules in front of the auth-callback
  path and `/health`; document that this replaces the invite-cap's implicit
  protection now that self-service sign-up is live.
- **Files/modules:** infra-only (Cloud Armor policy), no application code.
- **Tests:** a simulated burst from one source is throttled without
  affecting a concurrent request from a different source.
- **Stopping condition:** the throttling test passes against the deployed
  policy.
- **Completion evidence:** the policy definition committed to the runbook;
  the throttling-test transcript (request timestamps/status codes).

### Day 11 — Robust per-user concurrency + admin kill switch
- **Objective:** enforce the per-user concurrent-active-session cap
  robustly (not just at creation time — re-check on resume paths too);
  implement the admin kill switch (Section G) as a real, tested control;
  add `owner_id` to `paid_actions` for attribution.
- **Files/modules:** `research_agent/leases.py`/`admission.py` call sites,
  `research_agent/telemetry.py` (additive column), a new admin-only
  toggle endpoint or config flag.
- **Tests:** a per-user cap-exceeded attempt is rejected; the kill switch,
  once flipped, rejects every paid action with a clean 503 and no provider
  call (tripwire test); flipping it back restores normal operation.
- **Stopping condition:** all three tests pass.
- **Completion evidence:** test output; a manual kill-switch drill
  transcript (flip on, confirm 503+no-provider-call, flip off, confirm
  recovery).

### Day 12 — Connection-pool tuning under load
- **Objective:** load-test the Cloud SQL connection pool under 2–3× the
  Day-7 concurrency; tune pool size and statement timeouts; confirm no
  connection exhaustion.
- **Files/modules:** the Postgres connection-pool configuration from Day 2.
- **Tests:** a load-test script (not part of `pytest`) exercising the pool
  under sustained concurrent ownership reads/writes.
- **Stopping condition:** the load test completes with no connection
  exhaustion and stable p95 latency at the target concurrency.
- **Completion evidence:** the load-test results (connections used, p95,
  error rate) recorded in the runbook.

### Day 13 — Rollback rehearsal with real data present
- **Objective:** repeat the Day-6 rollback drill, but this time with real
  (test) user accounts and owned sessions present, to confirm rollback does
  not corrupt or leak data across the version boundary; begin incrementally
  raising the approval-gate threshold rather than removing it outright.
- **Files/modules:** none — operational drill.
- **Tests:** post-rollback, run the Section K ownership-isolation tests
  again against the rolled-back version.
- **Stopping condition:** the post-rollback isolation tests pass; the
  approval-gate change (e.g. raised limit) is documented as a deliberate,
  reversible config change.
- **Completion evidence:** the rollback-with-data-present drill transcript;
  the post-rollback test output; the gate-change entry in the runbook.

### Day 14 — Multi-user testing, documentation, presentation
- **Objective:** a mixed real + synthetic 4–8 concurrent-user run; finalize
  documentation; prepare the mentor presentation; write the final
  residual-risk report explicitly naming what remains before Tier 3.
- **Files/modules:** documentation only.
- **Tests:** the full Section K matrix, re-run at the new concurrency
  target.
- **Stopping condition:** the matrix passes at the new concurrency; the
  residual-risk report explicitly enumerates the Tier-3 gap (Part L).
- **Completion evidence:** the load-test metrics at 4–8 concurrent users;
  the final residual-risk report file.

---

## K. Acceptance and rollback plan

1. Registration/sign-in/sign-out round trip via Firebase (Google provider).
2. Invalid, expired, and forged identity tokens → 401 with a stable reason
   code, never a stack trace or partial success.
3. User A's `GET /curation/reviews` / `GET /library` return only A's rows —
   never B's, under both Tier 1 (approval-gated) and Tier 2 (self-service)
   configurations.
4. Cross-owner access on every route in the Part F matrix → **404**.
5. A rejected cross-owner mutation causes **zero** state change, verified by
   re-reading B's resource afterward.
6. `chat/stream` and `report*/stream` reject ownership **before** the
   `StreamingResponse` is constructed — proven with a hard provider-call
   tripwire, not log inspection.
7. `report/export` (markdown/pdf/docx) enforces ownership identically to the
   read path.
8. A rejected/unauthorized request never reaches a provider call — proven
   with the same tripwire as #6, applied to every guarded route, not only
   streaming ones.
9. Per-user budget rejections never affect an unrelated user's admission
   decision (distinct `subject_id`).
10. The global emergency ceiling still triggers under simulated multi-user
    burst traffic, and the admin kill switch (Tier 2) independently halts
    all paid actions on demand.
11. Two users generating reports simultaneously do not starve `/health`
    (polled every 5s throughout the run, must stay fast and 200).
12. A container restart preserves both Postgres (users/ownership) and disk
    (Chroma/checkpoints) state.
13. Backup and restore (Cloud SQL backup/PITR + `scripts/data_backup.py`)
    reproduce byte-identical/row-identical state in a scratch environment.
14. No secret — including no Firebase service-account key, which must not
    exist anywhere in this architecture (Correction 2) — appears in Git
    history, Docker image layers, or the frontend production bundle. (A
    Firebase *client* config/API key in the bundle is public-by-design and
    is not a secret; confirm specifically that no *server*-side credential
    leaks.)
15. Chroma remains consistent (no corruption, correct read-your-writes)
    under the Day-7 and Day-14 bounded concurrent load runs.
16. Deployment rollback (image-digest swap) works, including the Day-13
    rehearsal with real test-user data present and a post-rollback
    isolation-test re-run.
17. The Correction-4 consistency tests (creation/deletion ordering, both
    orphan classes, reconciliation-job correctness, concurrent-create
    independence) all pass — treated as a first-class acceptance item, not
    an implementation detail, since a failure here is a cross-user data
    exposure risk.

**Rollback:** images deployed by immutable digest, last 3 kept; app
rollback is a digest swap. Data rollback: Cloud SQL PITR for
users/ownership, disk-snapshot restore for Chroma/checkpoints — taken
before every deploy. An auth-broken deployment is fixed by restoring the
previous digest, never by disabling the identity gate.

---

## L. Explicit deferred work — scope classification

| Item | Classification |
|---|---|
| PostgreSQL for users/ownership/saved-searches | Required before Tier 1 |
| Firebase Authentication (Google provider only) | Required before Tier 1 |
| ADC-based credential handling (no service-account key) | Required before Tier 1 — a key file must never be introduced at any tier |
| AnyIO threadpool ceiling + async `/health` | Required before Tier 1 |
| Creation/deletion consistency model + reconciliation job (Correction 4) | Required before Tier 1 |
| Approval gate on new accounts | Required before Tier 1; deliberately relaxed, not removed, at Tier 2 |
| Full account-deletion lifecycle (soft delete + async purge) | Required before Tier 2 |
| IP-level throttling / Cloud Armor | Required before Tier 2 |
| Signup-abuse protection beyond the approval gate | Required before Tier 2 |
| Admin kill switch | Required before Tier 2 (a manual DB flip is an acceptable Tier 1 stopgap) |
| Connection-pool load tuning | Required before Tier 2 |
| Admin dashboard | Required before Tier 3; manual DB queries/scripts suffice through Tier 2 |
| `LangGraph PostgresSaver` migration | Deferred until a Correction-5 trigger actually fires (Tier 3 only) |
| Vector-store migration (pgvector / managed / Chroma server mode) | Deferred until the same Tier-3 trigger as `PostgresSaver` |
| Horizontal scaling / multi-instance app | Tier 3 only; not attempted by either schedule here |
| CI pipeline | Required before Tier 3; a manual pre-deploy checklist is acceptable through Tier 2 |
| Billing | Deferred — usage limits, not billing, govern cost through Tier 2 |
| `api.py` restructuring | Safe to defer indefinitely — hygiene only, no user-facing risk |
| Async route conversion | Safe to defer through Tier 2; revisit only if the threadpool ceiling (already mitigated) proves insufficient under real load |
| PDF ingestion | Safe to defer — unrelated feature work |
| Research Lanes | Safe to defer — keep flag off through Tier 2 |
| Policy C keyword filtering | Safe to defer — keep flag off through Tier 2 |
| Email/password authentication | Safe to defer — Google-only sign-in is sufficient through Tier 2 |
| Full historical-session (772 MB) migration | Safe to defer indefinitely — cold, read-only archive, never converted |
| Google Identity Platform upgrade (multi-tenancy/SAML) | Safe to defer indefinitely — only relevant if enterprise/SAML customers are ever targeted (Correction 1) |

---

## M. Final verdict and first implementation checkpoint

**Verdict: safe only as an approval-gated controlled beta (Tier 1) in seven
days.** The Tier-1 schedule delivers real Firebase authentication with no
service-account-key anti-pattern, real cross-store-consistent ownership
enforcement (Correction 4) across the full route matrix, real per-user and
global cost governance, and a genuine self-service *sign-up* path for the
mentor — gated by an explicit approval flag, one VM, and a tester cap. It
must be presented to the mentor as exactly that: an approval-gated
controlled beta, not a self-service public beta and not unrestricted
production.

Reaching **Tier 2 (self-service public beta)** requires the full 10–14 day
schedule in Section J — completing the account-deletion lifecycle, making
IP throttling and signup-abuse protection load-bearing, tuning the
connection pool under real load, and rehearsing rollback with real data
present. Even Tier 2 is explicitly not Tier 3: it stays on one VM with
`SqliteSaver`, and does not claim horizontal scalability.

**Tier 3 (unrestricted, scalable production)** is out of scope for both
schedules and should not be attempted until a concrete, evidenced need for
more than one application instance appears — at which point `PostgresSaver`
(Correction 5) and a Chroma/vector-store migration become mandatory
prerequisites, not optional hardening.

**First implementation checkpoint after this review: Day 1 of Section I** —
push the H5 commits, freeze this document as scope, create the Firebase
project with only the Google provider enabled, convert `/health` to
`async def`, and set the AnyIO thread-pool ceiling explicitly. Nothing in
Day 1 touches ownership, infrastructure, or credentials, and every step is
independently reversible.
