# Firebase Authentication — setup

`AUTH_MODE=firebase` gates the backend on Firebase ID tokens (public
multi-user identity). This document is the manual setup a human must do
in the Firebase / Google Cloud console — the application code creates no
Firebase resources and invents no project ID.

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
  separate administrative action (a later phase); `/me` works for an
  unapproved account so the frontend can render an "awaiting approval"
  state.
