-- Add missing tables for auth events, connector accounts, and usage metrics

-- ============================================================================
-- Auth Events (Google sign-in tracking)
-- ============================================================================

CREATE TABLE auth_events (
    id TEXT PRIMARY KEY,
    at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    kind TEXT NOT NULL,
    detail JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ix_auth_events_at ON auth_events(at DESC);

-- ============================================================================
-- Connector Accounts (Google account metadata)
-- ============================================================================

CREATE TABLE connector_accounts (
    connector TEXT NOT NULL,
    sub TEXT NOT NULL,
    email TEXT,
    name TEXT,
    picture TEXT,
    scopes TEXT[] DEFAULT ARRAY[]::TEXT[],
    token_expires_at TIMESTAMPTZ,
    last_sync_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (connector, sub)
);

-- ============================================================================
-- Usage Metrics (token tracking and feature metrics)
-- ============================================================================

CREATE TABLE usage (
    id TEXT PRIMARY KEY,
    feature TEXT NOT NULL,
    day DATE NOT NULL,
    tokens INTEGER DEFAULT 0,
    count INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(feature, day)
);

CREATE INDEX ix_usage_day ON usage(day DESC);
