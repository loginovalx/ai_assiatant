-- Две вещи: отметка об отправленном напоминании и идентификатор строки импортированного CSV.
--
-- notified_at — когда бот отправил напоминание пользователю. NULL — ещё не отправляли.
-- Нужна, чтобы после перезапуска сервиса напоминание не пришло повторно: источник
-- истины в базе, а не в памяти процесса.
--
-- external_id — значение колонки id из загруженного CSV. Уникален в пределах пользователя,
-- поэтому повторная загрузка того же файла не создаёт дублей (ON CONFLICT DO NOTHING).
--
-- На пустом volume применяется автоматически после 002. На уже работающей базе — один раз вручную:
--   docker compose exec -T postgres psql -U assistant -d assistant < db/migrations/003_notifications_and_import.sql

ALTER TABLE events ADD COLUMN IF NOT EXISTS notified_at timestamptz;
ALTER TABLE events ADD COLUMN IF NOT EXISTS external_id text;

-- Повторный импорт того же файла не плодит дубли. Частичный индекс: события,
-- созданные голосом, external_id не имеют, и уникальность их не касается.
CREATE UNIQUE INDEX IF NOT EXISTS events_user_external_idx
    ON events (user_id, external_id) WHERE external_id IS NOT NULL;

-- Планировщик раз в полминуты ищет наступившие и ещё не отправленные напоминания.
CREATE INDEX IF NOT EXISTS events_due_idx
    ON events (start_at) WHERE notified_at IS NULL AND status = 'active';
