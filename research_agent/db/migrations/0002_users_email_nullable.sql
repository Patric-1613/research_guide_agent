-- Day 3: a Firebase ID token is not guaranteed to carry an `email`
-- claim (identity providers vary; only Google is enabled today and it
-- always provides one, but the schema must not depend on that). email
-- is mutable metadata, never the ownership key -- `id` (and the unique
-- `firebase_uid`) already carry identity -- so a null email is a valid
-- user row, not a data-integrity problem.
ALTER TABLE users ALTER COLUMN email DROP NOT NULL;
