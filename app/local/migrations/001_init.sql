-- Initial schema for voxscribe-engine

CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- Meetings
-- ============================================================================

CREATE TABLE meetings (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    context TEXT,
    status TEXT NOT NULL DEFAULT 'scheduled',
    scheduled_start TIMESTAMPTZ,
    scheduled_end TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    calendar_event_id TEXT,
    reminder_enabled BOOLEAN DEFAULT false,
    enhance_style TEXT DEFAULT 'standard',
    auto_stop_minutes INTEGER,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ix_meetings_status ON meetings(status);
CREATE INDEX ix_meetings_created ON meetings(created_at DESC);

-- ============================================================================
-- Transcript Segments
-- ============================================================================

CREATE TABLE transcript_segments (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    source TEXT NOT NULL,
    speaker TEXT,
    text TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'provisional',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(meeting_id, source, start_ms, end_ms)
);

CREATE INDEX ix_transcript_meeting ON transcript_segments(meeting_id);

-- ============================================================================
-- Notes (Shorthand Blocks)
-- ============================================================================

CREATE TABLE notes_blocks (
    block_id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    offset_ms INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ix_notes_meeting ON notes_blocks(meeting_id);

-- ============================================================================
-- Enhanced Blocks
-- ============================================================================

CREATE TABLE enhanced_blocks (
    block_id TEXT PRIMARY KEY REFERENCES notes_blocks(block_id) ON DELETE CASCADE,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    citations JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ix_enhanced_meeting ON enhanced_blocks(meeting_id);

-- ============================================================================
-- Summaries
-- ============================================================================

CREATE TABLE summaries (
    meeting_id TEXT PRIMARY KEY REFERENCES meetings(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'none',
    overview TEXT,
    details JSONB DEFAULT '[]'::jsonb,
    generated_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- Sources (Knowledge Base)
-- ============================================================================

CREATE TABLE sources (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    uri TEXT NOT NULL,
    name TEXT NOT NULL,
    mime TEXT,
    parent_id TEXT REFERENCES sources(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'queued',
    error TEXT,
    imported_at TIMESTAMPTZ,
    chunk_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(kind, uri)
);

CREATE INDEX ix_sources_kind ON sources(kind);

-- ============================================================================
-- Source Chunks and Embeddings
-- ============================================================================

CREATE TABLE source_chunks (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    text_tsvector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', text)
    ) STORED,
    locator TEXT,
    embedding vector(768),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, ordinal)
);

CREATE INDEX ix_source_chunks_source ON source_chunks(source_id);
CREATE INDEX ix_source_chunks_text ON source_chunks USING GIN(text_tsvector);
CREATE INDEX ix_source_chunks_embedding ON source_chunks USING HNSW(embedding vector_cosine_ops);

-- ============================================================================
-- Labels
-- ============================================================================

CREATE TABLE labels (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    color TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- Source-Label Association
-- ============================================================================

CREATE TABLE source_labels (
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    label_id TEXT NOT NULL REFERENCES labels(id) ON DELETE CASCADE,
    PRIMARY KEY (source_id, label_id)
);

-- ============================================================================
-- Meeting Attachments
-- ============================================================================

CREATE TABLE meeting_attachments (
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    PRIMARY KEY (meeting_id, source_id)
);

CREATE INDEX ix_attachments_meeting ON meeting_attachments(meeting_id);

-- ============================================================================
-- Chat
-- ============================================================================

CREATE TABLE chat_threads (
    id TEXT PRIMARY KEY,
    label_ids TEXT[] DEFAULT ARRAY[]::TEXT[],
    meeting_id TEXT,
    source_ids TEXT[] DEFAULT ARRAY[]::TEXT[],
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE chat_messages (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    citations JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- Calendar Events
-- ============================================================================

CREATE TABLE calendar_events (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    "start" TIMESTAMPTZ NOT NULL,
    "end" TIMESTAMPTZ NOT NULL,
    attendees TEXT[] DEFAULT ARRAY[]::TEXT[],
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- Settings
-- ============================================================================

CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL
);

-- ============================================================================
-- Meta (for tracking embedder dimension)
-- ============================================================================

CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
