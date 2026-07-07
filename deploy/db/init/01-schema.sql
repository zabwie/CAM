-- Cam PostgreSQL/TimescaleDB schema
-- ponytail: basic schema. TimescaleDB hypertables added when volume justifies it.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts BIGINT NOT NULL,
    device_id TEXT NOT NULL,
    type TEXT NOT NULL,
    plugin_id TEXT NOT NULL,
    severity INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 1.0,
    source INTEGER NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}',
    clip_id UUID,
    twin_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_device ON events(device_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type, ts DESC);

CREATE TABLE IF NOT EXISTS devices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL DEFAULT '',
    lat DOUBLE PRECISION,
    lng DOUBLE PRECISION,
    hardware_model TEXT,
    os_version TEXT,
    firmware TEXT,
    status TEXT NOT NULL DEFAULT 'offline',
    last_seen BIGINT,
    config JSONB NOT NULL DEFAULT '{}',
    capabilities JSONB NOT NULL DEFAULT '{}',
    health JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS hourly_aggregates (
    device_id UUID NOT NULL,
    hour TIMESTAMPTZ NOT NULL,
    vehicle_count INTEGER NOT NULL DEFAULT 0,
    vehicle_counts JSONB NOT NULL DEFAULT '{}',
    avg_speed REAL,
    p50_speed REAL,
    p95_speed REAL,
    max_speed REAL,
    event_counts JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (device_id, hour)
);

SELECT create_hypertable('events', 'ts', if_not_exists => TRUE);
SELECT create_hypertable('hourly_aggregates', 'hour', if_not_exists => TRUE);
