# Datenbankschema-Dokumentation

## Überblick

Die YouTube Video Aggregation Pipeline verwendet Supabase (PostgreSQL) als primäre Datenbank mit Unterstützung für Vektor-Embeddings. Das Schema ist optimiert für sowohl traditionelle relationale Abfragen als auch moderne Vektor-Ähnlichkeitssuchen.

## Tabellen-Struktur

### 1. Videos Tabelle

**Zweck**: Speicherung von YouTube-Video-Metadaten

```sql
CREATE TABLE videos (
    video_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    channel_id TEXT NOT NULL,
    channel_title TEXT NOT NULL,
    published_at TIMESTAMP WITH TIME ZONE,
    duration INTEGER, -- Dauer in Sekunden
    view_count BIGINT DEFAULT 0,
    like_count INTEGER DEFAULT 0,
    comment_count INTEGER DEFAULT 0,
    tags TEXT[], -- Array von Tags
    category_id INTEGER,
    language TEXT DEFAULT 'unknown',
    relevance_score DECIMAL(3,2), -- KI-Klassifizierungs-Score
    is_relevant BOOLEAN DEFAULT FALSE,
    indexed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### Feld-Beschreibungen

| Feld | Typ | Beschreibung | Beispiel |
|------|-----|--------------|----------|
| `video_id` | TEXT | YouTube Video-ID (Primary Key) | "dQw4w9WgXcQ" |
| `title` | TEXT | Video-Titel | "Machine Learning Tutorial" |
| `description` | TEXT | Video-Beschreibung | "Learn the basics of..." |
| `channel_id` | TEXT | YouTube Kanal-ID | "UC_x5XG1OV2P6uZZ5FSM9Ttw" |
| `channel_title` | TEXT | Kanal-Name | "Google for Developers" |
| `published_at` | TIMESTAMP | Veröffentlichungsdatum | "2024-01-15T10:00:00Z" |
| `duration` | INTEGER | Video-Länge in Sekunden | 1800 |
| `view_count` | BIGINT | Anzahl Aufrufe | 1500000 |
| `like_count` | INTEGER | Anzahl Likes | 25000 |
| `comment_count` | INTEGER | Anzahl Kommentare | 500 |
| `tags` | TEXT[] | Video-Tags | {"AI", "Tutorial", "Python"} |
| `category_id` | INTEGER | YouTube Kategorie-ID | 28 |
| `language` | TEXT | Sprache des Videos | "en" |
| `relevance_score` | DECIMAL | KI-Bewertung (0.0-1.0) | 0.85 |
| `is_relevant` | BOOLEAN | Relevanz-Flag | TRUE |
| `indexed_at` | TIMESTAMP | Zeitpunkt der Indexierung | "2024-01-15T12:00:00Z" |
| `updated_at` | TIMESTAMP | Letzte Aktualisierung | "2024-01-15T12:00:00Z" |

#### Indizes

```sql
-- Performance-Indizes
CREATE INDEX idx_videos_channel_id ON videos(channel_id);
CREATE INDEX idx_videos_published_at ON videos(published_at DESC);
CREATE INDEX idx_videos_relevance_score ON videos(relevance_score DESC);
CREATE INDEX idx_videos_is_relevant ON videos(is_relevant);
CREATE INDEX idx_videos_view_count ON videos(view_count DESC);

-- Volltext-Suche
CREATE INDEX idx_videos_title_fts ON videos USING gin(to_tsvector('english', title));
CREATE INDEX idx_videos_description_fts ON videos USING gin(to_tsvector('english', description));

-- Array-Suche für Tags
CREATE INDEX idx_videos_tags ON videos USING gin(tags);
```

### 2. Video Chunks Tabelle

**Zweck**: Speicherung von Text-Segmenten mit Vektor-Embeddings

```sql
CREATE TABLE video_chunks (
    id SERIAL PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    char_count INTEGER NOT NULL,
    start_time INTEGER, -- Start-Zeit im Video (Sekunden)
    end_time INTEGER,   -- End-Zeit im Video (Sekunden)
    embedding VECTOR(1536), -- OpenAI ada-002 Embedding
    indexed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Composite Unique Constraint
    UNIQUE(video_id, chunk_index)
);
```

#### Feld-Beschreibungen

| Feld | Typ | Beschreibung | Beispiel |
|------|-----|--------------|----------|
| `id` | SERIAL | Auto-increment Primary Key | 12345 |
| `video_id` | TEXT | Referenz zur Videos-Tabelle | "dQw4w9WgXcQ" |
| `chunk_index` | INTEGER | Chunk-Nummer im Video | 0, 1, 2, ... |
| `content` | TEXT | Text-Inhalt des Chunks | "In this section we discuss..." |
| `token_count` | INTEGER | Anzahl Tokens (OpenAI) | 150 |
| `char_count` | INTEGER | Anzahl Zeichen | 800 |
| `start_time` | INTEGER | Start-Zeit in Sekunden | 120 |
| `end_time` | INTEGER | End-Zeit in Sekunden | 180 |
| `embedding` | VECTOR(1536) | Vektor-Embedding | [0.1, -0.2, 0.3, ...] |
| `indexed_at` | TIMESTAMP | Indexierungs-Zeitpunkt | "2024-01-15T12:05:00Z" |

#### Indizes

```sql
-- Performance-Indizes
CREATE INDEX idx_chunks_video_id ON video_chunks(video_id);
CREATE INDEX idx_chunks_video_chunk ON video_chunks(video_id, chunk_index);
CREATE INDEX idx_chunks_token_count ON video_chunks(token_count);

-- Volltext-Suche
CREATE INDEX idx_chunks_content_fts ON video_chunks USING gin(to_tsvector('english', content));

-- Vektor-Ähnlichkeitssuche (erfordert pgvector Extension)
CREATE INDEX idx_chunks_embedding ON video_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### 3. Processing Stats Tabelle (Optional)

**Zweck**: Tracking von Pipeline-Ausführungen

```sql
CREATE TABLE processing_stats (
    id SERIAL PRIMARY KEY,
    run_id UUID DEFAULT gen_random_uuid(),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    total_urls INTEGER DEFAULT 0,
    metadata_fetched INTEGER DEFAULT 0,
    classified_relevant INTEGER DEFAULT 0,
    transcripts_extracted INTEGER DEFAULT 0,
    chunks_created INTEGER DEFAULT 0,
    embeddings_generated INTEGER DEFAULT 0,
    indexed_videos INTEGER DEFAULT 0,
    indexed_chunks INTEGER DEFAULT 0,
    errors INTEGER DEFAULT 0,
    error_details JSONB,
    processing_time_seconds DECIMAL(10,3),
    config_snapshot JSONB
);
```

## Vektor-Suche Setup

### pgvector Extension

```sql
-- Extension aktivieren
CREATE EXTENSION IF NOT EXISTS vector;

-- Vektor-Ähnlichkeitssuche Funktionen
-- Cosine Similarity
SELECT 
    vc.video_id,
    vc.content,
    1 - (vc.embedding <=> $1::vector) AS similarity
FROM video_chunks vc
ORDER BY vc.embedding <=> $1::vector
LIMIT 10;

-- Euclidean Distance
SELECT 
    vc.video_id,
    vc.content,
    vc.embedding <-> $1::vector AS distance
FROM video_chunks vc
ORDER BY vc.embedding <-> $1::vector
LIMIT 10;
```

### Embedding-Suche Optimierung

```sql
-- IVFFlat Index für große Datensätze
CREATE INDEX CONCURRENTLY idx_chunks_embedding_ivfflat 
ON video_chunks 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 1000);

-- HNSW Index für bessere Performance (PostgreSQL 14+)
CREATE INDEX CONCURRENTLY idx_chunks_embedding_hnsw 
ON video_chunks 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);
```

## Häufige Abfragen

### 1. Relevante Videos finden

```sql
SELECT 
    video_id,
    title,
    channel_title,
    relevance_score,
    view_count
FROM videos 
WHERE is_relevant = TRUE 
    AND relevance_score >= 0.8
ORDER BY relevance_score DESC, view_count DESC
LIMIT 20;
```

### 2. Semantische Textsuche

```sql
-- Kombinierte Vektor- und Volltext-Suche
WITH semantic_search AS (
    SELECT 
        video_id,
        content,
        chunk_index,
        1 - (embedding <=> $1::vector) AS similarity
    FROM video_chunks
    WHERE embedding <=> $1::vector < 0.3  -- Similarity threshold
    ORDER BY embedding <=> $1::vector
    LIMIT 50
),
text_search AS (
    SELECT 
        video_id,
        content,
        chunk_index,
        ts_rank(to_tsvector('english', content), plainto_tsquery('english', $2)) AS text_rank
    FROM video_chunks
    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $2)
)
SELECT DISTINCT
    v.title,
    v.channel_title,
    ss.content,
    ss.similarity,
    ts.text_rank
FROM semantic_search ss
JOIN videos v ON ss.video_id = v.video_id
LEFT JOIN text_search ts ON ss.video_id = ts.video_id AND ss.chunk_index = ts.chunk_index
ORDER BY 
    COALESCE(ss.similarity, 0) * 0.7 + COALESCE(ts.text_rank, 0) * 0.3 DESC
LIMIT 20;
```

### 3. Kanal-Analyse

```sql
SELECT 
    channel_title,
    COUNT(*) as video_count,
    AVG(relevance_score) as avg_relevance,
    SUM(view_count) as total_views,
    AVG(view_count) as avg_views,
    MAX(published_at) as latest_video
FROM videos 
WHERE is_relevant = TRUE
GROUP BY channel_id, channel_title
HAVING COUNT(*) >= 3
ORDER BY avg_relevance DESC, total_views DESC;
```

### 4. Trend-Analyse

```sql
-- Monatliche Trends
SELECT 
    DATE_TRUNC('month', published_at) as month,
    COUNT(*) as video_count,
    AVG(relevance_score) as avg_relevance,
    AVG(view_count) as avg_views
FROM videos 
WHERE is_relevant = TRUE 
    AND published_at >= NOW() - INTERVAL '12 months'
GROUP BY DATE_TRUNC('month', published_at)
ORDER BY month DESC;
```

### 5. Tag-Analyse

```sql
-- Häufigste Tags
SELECT 
    unnest(tags) as tag,
    COUNT(*) as frequency,
    AVG(relevance_score) as avg_relevance,
    AVG(view_count) as avg_views
FROM videos 
WHERE is_relevant = TRUE 
    AND tags IS NOT NULL
GROUP BY unnest(tags)
HAVING COUNT(*) >= 5
ORDER BY frequency DESC, avg_relevance DESC
LIMIT 50;
```

## Performance-Optimierung

### 1. Partitionierung

```sql
-- Partitionierung nach Datum für große Datensätze
CREATE TABLE videos_partitioned (
    LIKE videos INCLUDING ALL
) PARTITION BY RANGE (published_at);

-- Monatliche Partitionen
CREATE TABLE videos_2024_01 PARTITION OF videos_partitioned
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE videos_2024_02 PARTITION OF videos_partitioned
FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
```

### 2. Materialized Views

```sql
-- Aggregierte Statistiken
CREATE MATERIALIZED VIEW channel_stats AS
SELECT 
    channel_id,
    channel_title,
    COUNT(*) as video_count,
    AVG(relevance_score) as avg_relevance,
    SUM(view_count) as total_views,
    MAX(published_at) as latest_video,
    MIN(published_at) as first_video
FROM videos 
WHERE is_relevant = TRUE
GROUP BY channel_id, channel_title;

-- Index für Materialized View
CREATE INDEX idx_channel_stats_relevance ON channel_stats(avg_relevance DESC);

-- Refresh-Strategie
REFRESH MATERIALIZED VIEW CONCURRENTLY channel_stats;
```

### 3. Archivierung

```sql
-- Archivierung alter Daten
CREATE TABLE videos_archive (
    LIKE videos INCLUDING ALL
);

-- Archivierungs-Prozedur
CREATE OR REPLACE FUNCTION archive_old_videos(cutoff_date DATE)
RETURNS INTEGER AS $$
DECLARE
    archived_count INTEGER;
BEGIN
    -- Verschiebe alte Videos
    WITH archived AS (
        DELETE FROM videos 
        WHERE published_at < cutoff_date 
            AND is_relevant = FALSE
        RETURNING *
    )
    INSERT INTO videos_archive 
    SELECT * FROM archived;
    
    GET DIAGNOSTICS archived_count = ROW_COUNT;
    RETURN archived_count;
END;
$$ LANGUAGE plpgsql;
```

## Backup und Recovery

### 1. Backup-Strategie

```bash
# Vollständiges Backup
pg_dump -h localhost -U postgres -d youtube_pipeline > backup_full.sql

# Schema-only Backup
pg_dump -h localhost -U postgres -d youtube_pipeline --schema-only > backup_schema.sql

# Nur Daten
pg_dump -h localhost -U postgres -d youtube_pipeline --data-only > backup_data.sql

# Komprimiertes Backup
pg_dump -h localhost -U postgres -d youtube_pipeline -Fc > backup_compressed.dump
```

### 2. Point-in-Time Recovery

```sql
-- WAL-Archivierung aktivieren
archive_mode = on
archive_command = 'cp %p /path/to/archive/%f'
wal_level = replica
```

## Monitoring und Wartung

### 1. Tabellen-Statistiken

```sql
-- Tabellengröße
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Index-Nutzung
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

### 2. Wartungsaufgaben

```sql
-- Vacuum und Analyze
VACUUM ANALYZE videos;
VACUUM ANALYZE video_chunks;

-- Reindex für Vektor-Indizes
REINDEX INDEX CONCURRENTLY idx_chunks_embedding;

-- Statistiken aktualisieren
ANALYZE videos;
ANALYZE video_chunks;
```

## Migration und Versionierung

### Schema-Versionen

```sql
-- Schema-Versionstabelle
CREATE TABLE schema_versions (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Aktuelle Version
INSERT INTO schema_versions (version, description) 
VALUES (1, 'Initial schema with videos and video_chunks tables');
```

### Migration-Skripte

```sql
-- Migration v1 -> v2: Hinzufügen von Relevanz-Feldern
ALTER TABLE videos ADD COLUMN IF NOT EXISTS relevance_score DECIMAL(3,2);
ALTER TABLE videos ADD COLUMN IF NOT EXISTS is_relevant BOOLEAN DEFAULT FALSE;

-- Update Version
INSERT INTO schema_versions (version, description) 
VALUES (2, 'Added relevance scoring fields');
```

---

**Hinweis**: Dieses Schema ist optimiert für Supabase/PostgreSQL mit pgvector Extension. Für andere Datenbanken können Anpassungen erforderlich sein.