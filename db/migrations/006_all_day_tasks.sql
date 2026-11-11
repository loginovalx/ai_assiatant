ALTER TABLE events
    ADD COLUMN IF NOT EXISTS all_day boolean NOT NULL DEFAULT false
    CHECK (NOT all_day OR (type = 'task' AND start_at IS NOT NULL AND duration_minutes IS NULL));
