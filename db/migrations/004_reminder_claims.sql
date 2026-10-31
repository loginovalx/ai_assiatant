-- Short leases recover unsent reminders after a sender process crashes.
-- Apply to an existing database without dropping data.
ALTER TABLE events ADD COLUMN IF NOT EXISTS notification_claimed_at timestamptz;
ALTER TABLE events ADD COLUMN IF NOT EXISTS notification_claim_token uuid;
