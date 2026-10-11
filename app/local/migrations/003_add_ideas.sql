-- Ideas, projects, features and related tables for IdeaScribe

-- ============================================================================
-- Projects
-- ============================================================================

CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    parent_id TEXT REFERENCES projects(id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ix_projects_parent ON projects(parent_id);
CREATE INDEX ix_projects_created ON projects(created_at DESC);

-- ============================================================================
-- Features (cluster centers for graph view)
-- ============================================================================

CREATE TABLE features (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ix_features_project ON features(project_id);

-- ============================================================================
-- Ideas
-- ============================================================================

CREATE TABLE ideas (
    id TEXT PRIMARY KEY,
    meeting_id TEXT REFERENCES meetings(id) ON DELETE SET NULL,
    project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
    feature_id TEXT REFERENCES features(id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    summary TEXT,
    type TEXT NOT NULL DEFAULT 'new_feature',
    status TEXT NOT NULL DEFAULT 'raw',
    score INTEGER NOT NULL DEFAULT 0,
    rubric JSONB DEFAULT '{}'::jsonb,
    owner TEXT,
    originator TEXT,
    contributors TEXT[] DEFAULT ARRAY[]::TEXT[],
    challengers TEXT[] DEFAULT ARRAY[]::TEXT[],
    edited BOOLEAN DEFAULT false,
    evidence JSONB DEFAULT '[]'::jsonb,
    challenges JSONB DEFAULT '[]'::jsonb,
    gaps TEXT[] DEFAULT ARRAY[]::TEXT[],
    assumptions TEXT[] DEFAULT ARRAY[]::TEXT[],
    contradictions JSONB DEFAULT '[]'::jsonb,
    embedding vector(768),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ix_ideas_project ON ideas(project_id);
CREATE INDEX ix_ideas_feature ON ideas(feature_id);
CREATE INDEX ix_ideas_meeting ON ideas(meeting_id);
CREATE INDEX ix_ideas_status ON ideas(status);
CREATE INDEX ix_ideas_score ON ideas(score);
CREATE INDEX ix_ideas_created ON ideas(created_at DESC);
CREATE INDEX ix_ideas_title_text ON ideas USING GIN(
    to_tsvector('english', title || ' ' || COALESCE(summary, ''))
);
CREATE INDEX ix_ideas_embedding ON ideas USING HNSW(embedding vector_cosine_ops);

-- ============================================================================
-- Idea Edges (lineage and relationships)
-- ============================================================================

CREATE TABLE idea_edges (
    id TEXT PRIMARY KEY,
    from_id TEXT NOT NULL REFERENCES ideas(id) ON DELETE CASCADE,
    to_id TEXT NOT NULL REFERENCES ideas(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(from_id, to_id, kind)
);

CREATE INDEX ix_idea_edges_from ON idea_edges(from_id);
CREATE INDEX ix_idea_edges_to ON idea_edges(to_id);
CREATE INDEX ix_idea_edges_kind ON idea_edges(kind);

-- ============================================================================
-- Idea Activity Log
-- ============================================================================

CREATE TABLE idea_activity (
    id TEXT PRIMARY KEY,
    idea_id TEXT NOT NULL REFERENCES ideas(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    detail JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ix_idea_activity_idea ON idea_activity(idea_id);
CREATE INDEX ix_idea_activity_kind ON idea_activity(kind);

-- ============================================================================
-- Idea PRDs
-- ============================================================================

CREATE TABLE idea_prds (
    idea_id TEXT PRIMARY KEY REFERENCES ideas(id) ON DELETE CASCADE,
    markdown TEXT,
    generated_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'none'
);
