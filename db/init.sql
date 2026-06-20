CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id      SERIAL PRIMARY KEY,
    url         TEXT UNIQUE NOT NULL,
    title       TEXT,
    description TEXT,
    body        TEXT,
    image_url   TEXT,
    category    TEXT,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS text_chunks (
    chunk_id    SERIAL PRIMARY KEY,
    doc_id      INTEGER REFERENCES documents(doc_id),
    content     TEXT NOT NULL,
    position    INTEGER,
    word_count  INTEGER,
    norm        FLOAT,          -- ||d||: norma TF-IDF del chunk, precalculada por SPIMI
    -- tsvector para full-text nativo de PostgreSQL (comparación GIN, Fase 3).
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- Índice GIN para acelerar las consultas full-text (tsv @@ query).
CREATE INDEX IF NOT EXISTS idx_text_chunks_tsv ON text_chunks USING GIN (tsv);

CREATE TABLE IF NOT EXISTS image_chunks (
    chunk_id    SERIAL PRIMARY KEY,
    doc_id      INTEGER REFERENCES documents(doc_id),
    image_url   TEXT,
    patch_index INTEGER,
    descriptor  vector(128)
);

CREATE TABLE IF NOT EXISTS codebook_text (
    word_id     SERIAL PRIMARY KEY,
    word        TEXT UNIQUE NOT NULL,
    frequency   INTEGER
);

CREATE TABLE IF NOT EXISTS inverted_index_text (
    word_id     INTEGER REFERENCES codebook_text(word_id),
    chunk_id    INTEGER REFERENCES text_chunks(chunk_id),
    tf_idf      FLOAT,
    PRIMARY KEY (word_id, chunk_id)
);