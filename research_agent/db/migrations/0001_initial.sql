-- Day 2 (public multi-user deployment foundation, see
-- docs/plans/public-multi-user-deployment-review.md): the initial
-- ownership schema. Applied inside one transaction by
-- research_agent/db/migrations.py's run_migrations() -- either every
-- statement below succeeds, or none of them are persisted.
--
-- users.id / curation_owners.owner_id / saved_searches.owner_id are
-- UUIDs generated in Python (uuid.uuid4()), never a database-side
-- default -- see research_agent/db/ownership_repository.py. No
-- pgcrypto/uuid-ossp extension is required by this schema.

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    firebase_uid TEXT NOT NULL,
    email TEXT NOT NULL,
    display_name TEXT,
    approved BOOLEAN NOT NULL DEFAULT false,
    disabled BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- firebase_uid is the immutable provider identity; email is mutable
-- metadata and deliberately carries NO uniqueness constraint here (an
-- account-deletion/re-signup flow could otherwise collide on a freed
-- email address -- Day 2 does not implement that lifecycle yet, but the
-- schema must not need a later constraint change to support it).
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_firebase_uid ON users (firebase_uid);

CREATE TABLE IF NOT EXISTS curation_owners (
    session_id TEXT PRIMARY KEY,
    owner_id UUID NOT NULL REFERENCES users (id),
    topic TEXT NOT NULL,
    display_title TEXT,
    stage TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Listing one user's sessions newest-first.
CREATE INDEX IF NOT EXISTS idx_curation_owners_owner_created
    ON curation_owners (owner_id, created_at DESC);

-- Counting (or listing) one user's sessions by stage -- "active session"
-- count queries filter on both columns together.
CREATE INDEX IF NOT EXISTS idx_curation_owners_owner_stage
    ON curation_owners (owner_id, stage);

-- (Looking up a session's owner by session_id needs no separate index --
-- session_id is already the primary key.)

CREATE TABLE IF NOT EXISTS saved_searches (
    id BIGSERIAL PRIMARY KEY,
    owner_id UUID NOT NULL REFERENCES users (id),
    topic TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    paper_ids JSONB NOT NULL,
    scores JSONB NOT NULL,
    summary JSONB,
    web_articles JSONB,
    web_summary JSONB
);

-- Listing one owner's saved searches newest-first, with a stable
-- (created_at, id) tie-break -- same ordering contract
-- research_agent/storage.py's own list_searches() already establishes
-- for the SQLite side.
CREATE INDEX IF NOT EXISTS idx_saved_searches_owner_created
    ON saved_searches (owner_id, created_at DESC, id DESC);
