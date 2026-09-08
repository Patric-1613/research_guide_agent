# Firebase Authentication — setup

`AUTH_MODE=firebase` gates the backend on Firebase ID tokens (public
multi-user identity). This document is the manual setup a human must do
in the Firebase / Google Cloud console — the application code creates no
Firebase resources and invents no project ID.

> **Backend authorization for `AUTH_MODE=firebase` is enforced**
> (approval gate + per-user curation ownership + legacy-route lockout —
> see [`architecture.md`](architecture.md#authorization-firebase-multi-user-mode)).
> A public deployment still needs the remaining checkpoints from
> `docs/plans/public-multi-user-deployment-review.md`: the frontend
> sign-in flow, and the GCP/Cloud SQL infrastructure. There is also no
> self-service approval UI yet — a new account stays `approved = false`
> until an operator flips the flag with SQL. `firebase` mode runs
> correctly locally against the emulator today; the supported *deployed*
> configurations remain `AUTH_MODE=basic` and (local only)
> `AUTH_MODE=disabled` until that work lands.

Configuration contract: `research_agent/config/settings.py`'s
`get_auth_config()`. Verification boundary:
`research_agent/firebase_auth.py`. Identity resolution and the
`/me` endpoint: `research_agent/identity.py`,
`research_agent/api_app/routers/me.py`.

## What the backend needs

Exactly one value, and it is **not a secret**:

- `FIREBASE_PROJECT_ID` — the Google Cloud / Firebase project ID (6–30
  lowercase letters, digits and hyphens; starts with a letter). Used as
  the ID-token audience and issuer suffix.

Verification uses only that project ID plus Google's **public**
signing certificates (fetched over HTTPS and cached per Google's own
`Cache-Control` header). There is **no service-account key**: no
`GOOGLE_APPLICATION_CREDENTIALS`, no `serviceAccountKey.json`, no
`credentials.Certificate(...)`. Any future Admin-SDK-style call (token
revocation, custom claims — not built yet) must use Application Default
Credentials from the runtime's attached service account, never a
downloaded key file.

## Manual console steps (one time)

1. **Link Firebase to the existing target GCP project.** In the Firebase
   console, "Add project" → "Add Firebase to an existing Google Cloud
   project" and select the project the VM / Cloud SQL already live in.
   Do **not** create a new, separate project — IAM, billing and the
   project ID stay unified.

2. **Enable only Google as a sign-in provider.** Firebase console →
   Authentication → Sign-in method → enable **Google**. Set the project
   support email. **Leave every other provider disabled** —
   Email/Password, Email link, Phone, Anonymous, Apple, GitHub,
   Microsoft, and any SAML / OIDC provider. The backend does not reject
   tokens by provider, so enabling another provider silently widens who
   can sign in.

3. **Register authorized domains.** Authentication → Settings →
   Authorized domains. Add:
   - `localhost` (present by default) — local development.
   - the production web origin (e.g. `app.example.com`) — add when the
     domain is known; the frontend's Firebase sign-in redirect fails
     from an unlisted domain.

4. **Record the public web app config.** Project settings → General →
   "Your apps" → add a Web app if none exists → copy the config object
   (`apiKey`, `authDomain`, `projectId`, `appId`, …). These are **public
   client-side values** (they ship in the browser bundle by design) and
   are consumed by the Day-5 frontend, not by this backend. The backend
   needs only `projectId`, set as `FIREBASE_PROJECT_ID`.

5. **Do not download a service-account key.** Project settings → Service
   accounts has a "Generate new private key" button — never use it for
   this deployment.

## Local development without a real Firebase project

Use the Firebase Auth **emulator**. When `FIREBASE_AUTH_EMULATOR_HOST`
is set (e.g. `127.0.0.1:9099`), the backend accepts tokens minted by a
locally-run emulator instead of a real Firebase project. Emulator tokens
are **unsigned** (`alg: none`); the backend skips the signature check in
that mode only but still enforces issuer, audience, expiry, issued-at
and a valid `sub`. `get_auth_config()` **refuses to start** if
`FIREBASE_AUTH_EMULATOR_HOST` is set while `APP_ENV=production`.

The emulator ships with the Firebase CLI (`firebase-tools`, an npm
package) and needs a Java runtime. It is **not** bundled with this repo
and is not started by the test suite. To use it:

```
npm install -g firebase-tools           # one-time, needs Node + Java
firebase emulators:start --only auth --project <FIREBASE_PROJECT_ID>
```

Then run the backend with:

```
APP_ENV=local
AUTH_MODE=firebase
FIREBASE_PROJECT_ID=<any valid-format id, e.g. demo-local-1234>
FIREBASE_AUTH_EMULATOR_HOST=127.0.0.1:9099
DATABASE_URL=postgresql://...          # the users table still lives in PostgreSQL
```

## Runtime requirements for `AUTH_MODE=firebase`

- `DATABASE_URL` **must** be configured — every verified token is
  resolved to an internal `users` row in PostgreSQL, and `create_app()`
  refuses to start `firebase` mode without a database.
- Run migrations (`research_agent/db/migrations/`) against that database
  before starting the app — they never run automatically on startup.
- New accounts are created with `approved = false`. Approval is a
  separate administrative action (an operator runs
  `UPDATE users SET approved = true WHERE …`); there is no approval UI
  yet. `/me` works for an unapproved account so the frontend can render
  an "awaiting approval" state.

## Authorization

Full detail: `docs/architecture.md`, "Authorization (Firebase
multi-user mode)". In `firebase` mode:

- **Product routes require an approved account.** An authenticated but
  unapproved (or `disabled`) account gets a generic `403` on every
  curation/report/lane route. `GET /me` is the deliberate exception — it
  works for any authenticated account so the frontend can show its
  status.
- **Curation sessions are scoped to their PostgreSQL owner.** Every
  `/curation/{session_id}` route (reads, mutations, both chat streams,
  both report streams, export, activate, delete) checks the
  `curation_owners` row against the caller's internal user id first. A
  session owned by someone else and one that does not exist return the
  **same** generic `404` — ownership is never disclosed. The check runs
  before the checkpointer is opened, before any stream, lease, or
  provider call.
- **`GET /curation/reviews` returns only the caller's own sessions.**
- **The legacy shared search family** (`/search`, `/library`,
  `/summarize`, `/chat`, `/export`) is **unavailable** (`403`) in
  `firebase` mode — it has no per-user scoping.
- Configuration or ownership-database failure → generic `503`, never a
  fall-through and never a fallback to shared storage.
- `basic` and `disabled` modes are unaffected: none of the above runs,
  and the single shared workspace behaves exactly as before.

## What verification does and does not check

`research_agent/firebase_auth.py` verifies the ID token's RS256
signature (against Google's public `securetoken@system` certificates),
the issuer, the audience (`FIREBASE_PROJECT_ID`), expiry and issued-at
(300 s clock skew), and a usable `sub`. Firebase *custom* tokens,
session cookies and tokens minted for a different project are rejected
by the issuer / signing-key check.

**Token revocation is intentionally not checked.** Verifying whether a
specific ID token has been revoked (password reset, "sign out of all
sessions", explicit revocation) needs an Admin-SDK-style call against
Google, which needs Application Default Credentials from the runtime's
attached service account — deliberately out of scope for this phase.
The compensating control is `users.disabled`: **every** request
re-resolves the token to its internal `users` row and is denied if that
row is `disabled` (identity is never cached between requests). So the
operator's lever for immediately locking out a specific person is
`UPDATE users SET disabled = true` — it takes effect on that account's
very next request, regardless of how long their existing ID token
remains otherwise valid (at most one hour). Revocation checking can be
added later without changing this contract.

## Failure modes

- A **bad, expired, malformed, wrong-project or missing** token → `401`
  with a single generic body. The verifier's own message is never
  returned or logged.
- A failure to **fetch Google's signing certificates** (network outage,
  5xx from Google's cert endpoint) → `503`
  (`reason_code: identity_store_unavailable`), identical in shape to a
  PostgreSQL outage. An inability to *complete* verification is treated
  as infrastructure unavailability, never as an invalid credential —
  the same fail-closed posture the admission and lease stores use. The
  request is still denied and the route never runs.
