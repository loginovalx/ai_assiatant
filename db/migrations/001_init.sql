-- Схема хранения задач, напоминаний и встреч.
-- Postgres-контейнер выполняет файлы из этой папки один раз, при первом старте на пустом volume.
-- Для существующей базы: `python -m scripts.migrate` (без удаления данных).

CREATE EXTENSION IF NOT EXISTS vector;   -- эмбеддинги для поиска по смыслу
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- нечёткий поиск по опечаткам

CREATE TABLE IF NOT EXISTS events (
    id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     text        NOT NULL,
    type        text        NOT NULL CHECK (type IN ('task', 'meeting', 'reminder')),
    name        text        NOT NULL CHECK (length(name) BETWEEN 1 AND 200),
    description text        NOT NULL DEFAULT '',
    start_at    timestamptz,
    status      text        NOT NULL DEFAULT 'active'
                            CHECK (status IN ('active', 'done', 'cancelled')),
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now(),

    -- Размерность модели GigaChat "Embeddings". Пусто, если эмбеддинг не удалось получить.
    embedding   vector(1024),

    -- Поля для поиска считает сама база, приложению о них думать не нужно.
    search_text text GENERATED ALWAYS AS (name || ' ' || description) STORED,
    search_tsv  tsvector GENERATED ALWAYS AS (
        to_tsvector('russian', name || ' ' || description)
    ) STORED,

    -- Задача может быть без даты («купить билеты»), встреча и напоминание — нет.
    CONSTRAINT timed_types_need_start CHECK (type = 'task' OR start_at IS NOT NULL)
);

-- «Что у меня сегодня» = фильтр по пользователю и диапазону дат.
CREATE INDEX IF NOT EXISTS events_user_start_idx ON events (user_id, start_at);
CREATE INDEX IF NOT EXISTS events_tsv_idx  ON events USING gin (search_tsv);
CREATE INDEX IF NOT EXISTS events_trgm_idx ON events USING gin (search_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS events_embedding_idx ON events USING hnsw (embedding vector_cosine_ops);

CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS events_updated_at ON events;
CREATE TRIGGER events_updated_at
    BEFORE UPDATE ON events
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
