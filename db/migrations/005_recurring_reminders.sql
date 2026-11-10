CREATE TABLE IF NOT EXISTS reminder_series (
    id uuid PRIMARY KEY,
    user_id text NOT NULL,
    name text NOT NULL CHECK (length(name) BETWEEN 1 AND 200),
    weekday smallint CHECK (weekday BETWEEN 0 AND 6),
    clock time NOT NULL,
    timezone text NOT NULL,
    starts_at timestamptz NOT NULL,
    generated_until date NOT NULL,
    revision uuid NOT NULL,
    active boolean NOT NULL DEFAULT true
);

ALTER TABLE events ADD COLUMN IF NOT EXISTS reminder_series_id uuid REFERENCES reminder_series(id);
ALTER TABLE events ADD COLUMN IF NOT EXISTS occurrence_at timestamptz;
CREATE UNIQUE INDEX IF NOT EXISTS events_series_occurrence_idx
    ON events (reminder_series_id, occurrence_at);
