import Database from "better-sqlite3";
import path from "path";
import fs from "fs";
import { config } from "./config.js";
import type {
  Event,
  Vehicle,
  Device,
  Calibration,
  Plugin,
  Alert,
  Clip,
  Telemetry,
  HourlyAggregate,
  TwinState,
} from "./types.js";

// ponytail: SQLite for v1. Swap for TimescaleDB/PostgreSQL when multi-server or high volume needed.
const dbDir = path.dirname(config.dbPath);
if (!fs.existsSync(dbDir)) fs.mkdirSync(dbDir, { recursive: true });

const db = new Database(config.dbPath);
db.pragma("journal_mode = WAL");
db.pragma("foreign_keys = ON");

export function initSchema(): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS events (
      id TEXT PRIMARY KEY,
      ts INTEGER NOT NULL,
      device_id TEXT NOT NULL,
      type TEXT NOT NULL,
      plugin_id TEXT NOT NULL,
      severity INTEGER NOT NULL DEFAULT 0,
      confidence REAL NOT NULL DEFAULT 1.0,
      source INTEGER NOT NULL DEFAULT 0,
      metadata TEXT NOT NULL DEFAULT '{}',
      clip_id TEXT,
      twin_path TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC);
    CREATE INDEX IF NOT EXISTS idx_events_device ON events(device_id, ts DESC);
    CREATE INDEX IF NOT EXISTS idx_events_type ON events(type, ts DESC);

    CREATE TABLE IF NOT EXISTS vehicles (
      id TEXT PRIMARY KEY,
      event_id TEXT NOT NULL REFERENCES events(id),
      track_id INTEGER NOT NULL,
      vehicle_class TEXT NOT NULL DEFAULT 'unknown',
      speed REAL,
      direction REAL,
      lane INTEGER,
      trajectory TEXT NOT NULL DEFAULT '[]',
      features TEXT NOT NULL DEFAULT '{}'
    );

    CREATE TABLE IF NOT EXISTS devices (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL DEFAULT '',
      lat REAL,
      lng REAL,
      hardware_model TEXT,
      os_version TEXT,
      firmware TEXT,
      status TEXT NOT NULL DEFAULT 'offline',
      last_seen INTEGER,
      config TEXT NOT NULL DEFAULT '{}',
      capabilities TEXT NOT NULL DEFAULT '{}',
      health TEXT NOT NULL DEFAULT '{}'
    );

    CREATE TABLE IF NOT EXISTS calibrations (
      id TEXT PRIMARY KEY,
      device_id TEXT NOT NULL REFERENCES devices(id),
      version INTEGER NOT NULL DEFAULT 1,
      homography TEXT NOT NULL DEFAULT '[]',
      lanes TEXT NOT NULL DEFAULT '{}',
      zones TEXT NOT NULL DEFAULT '{}',
      confidence REAL NOT NULL DEFAULT 0.0,
      created_by TEXT NOT NULL DEFAULT 'auto',
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      active INTEGER NOT NULL DEFAULT 1
    );

    CREATE TABLE IF NOT EXISTS plugins (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      version TEXT NOT NULL DEFAULT '0.1.0',
      tier TEXT NOT NULL DEFAULT 'starter',
      enabled INTEGER NOT NULL DEFAULT 1,
      config TEXT NOT NULL DEFAULT '{}'
    );

    CREATE TABLE IF NOT EXISTS alerts (
      id TEXT PRIMARY KEY,
      event_id TEXT REFERENCES events(id),
      rule_id TEXT NOT NULL DEFAULT '',
      title TEXT NOT NULL,
      message TEXT NOT NULL,
      severity TEXT NOT NULL DEFAULT 'info',
      acknowledged INTEGER NOT NULL DEFAULT 0,
      created_at INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS clips (
      id TEXT PRIMARY KEY,
      device_id TEXT NOT NULL,
      timestamp INTEGER NOT NULL,
      duration INTEGER NOT NULL DEFAULT 0,
      size_bytes INTEGER NOT NULL DEFAULT 0,
      storage_path TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT 'uploading'
    );

    CREATE TABLE IF NOT EXISTS telemetry (
      device_id TEXT NOT NULL,
      timestamp INTEGER NOT NULL,
      cpu_usage REAL,
      gpu_usage REAL,
      memory_usage REAL,
      temperature REAL,
      uptime INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_telemetry_device ON telemetry(device_id, timestamp DESC);

    CREATE TABLE IF NOT EXISTS hourly_aggregates (
      device_id TEXT NOT NULL,
      hour TEXT NOT NULL,
      vehicle_count INTEGER NOT NULL DEFAULT 0,
      vehicle_counts TEXT NOT NULL DEFAULT '{}',
      avg_speed REAL,
      p50_speed REAL,
      p95_speed REAL,
      max_speed REAL,
      event_counts TEXT NOT NULL DEFAULT '{}',
      PRIMARY KEY (device_id, hour)
    );

    CREATE TABLE IF NOT EXISTS twin_state (
      id TEXT PRIMARY KEY,
      twin_path TEXT NOT NULL UNIQUE,
      state TEXT NOT NULL DEFAULT '{}',
      confidence REAL NOT NULL DEFAULT 1.0,
      recorded_at INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      email TEXT UNIQUE NOT NULL,
      password_hash TEXT NOT NULL,
      role TEXT NOT NULL DEFAULT 'viewer',
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
  `);
}

// ── Event queries ──
export function insertEvent(event: Event): void {
  db.prepare(
    `INSERT INTO events (id, ts, device_id, type, plugin_id, severity, confidence, source, metadata, clip_id, twin_path)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
  ).run(
    event.id, event.ts, event.device_id, event.type, event.plugin_id,
    event.severity, event.confidence, event.source,
    JSON.stringify(event.metadata), event.clip_id ?? null, event.twin_path ?? null
  );
}

export function queryEvents(
  filters: { device_id?: string; type?: string; plugin_id?: string; severity?: number; since?: number; limit?: number }
): Event[] {
  const clauses: string[] = [];
  const params: unknown[] = [];
  if (filters.device_id) { clauses.push("device_id = ?"); params.push(filters.device_id); }
  if (filters.type) { clauses.push("type = ?"); params.push(filters.type); }
  if (filters.plugin_id) { clauses.push("plugin_id = ?"); params.push(filters.plugin_id); }
  if (filters.severity !== undefined) { clauses.push("severity = ?"); params.push(filters.severity); }
  if (filters.since) { clauses.push("ts >= ?"); params.push(filters.since); }
  const where = clauses.length ? "WHERE " + clauses.join(" AND ") : "";
  const limit = Math.min(filters.limit ?? 100, 1000);
  return db.prepare(`SELECT * FROM events ${where} ORDER BY ts DESC LIMIT ?`).all(...params, limit) as Event[];
}

// ── Device queries ──
export function upsertDevice(device: Device): void {
  db.prepare(
    `INSERT INTO devices (id, name, lat, lng, hardware_model, os_version, firmware, status, last_seen, config, capabilities, health)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET
       name=excluded.name, status=excluded.status, last_seen=excluded.last_seen,
       firmware=excluded.firmware, os_version=excluded.os_version,
       config=excluded.config, health=excluded.health,
       capabilities=excluded.capabilities`
  ).run(
    device.id, device.name, device.location?.lat ?? null, device.location?.lng ?? null,
    device.hardware_model, device.os_version, device.firmware,
    device.status, device.last_seen,
    JSON.stringify(device.config), JSON.stringify(device.capabilities), JSON.stringify(device.health)
  );
}

export function getDevices(): Device[] {
  return db.prepare("SELECT * FROM devices ORDER BY last_seen DESC").all() as Device[];
}

export function getDevice(id: string): Device | undefined {
  return db.prepare("SELECT * FROM devices WHERE id = ?").get(id) as Device | undefined;
}

// ── Other queries ──
export function getCalibrations(device_id?: string): Calibration[] {
  if (device_id) return db.prepare("SELECT * FROM calibrations WHERE device_id = ? ORDER BY version DESC").all(device_id) as Calibration[];
  return db.prepare("SELECT * FROM calibrations ORDER BY created_at DESC").all() as Calibration[];
}

export function getPlugins(): Plugin[] {
  return db.prepare("SELECT * FROM plugins ORDER BY id").all() as Plugin[];
}

export function upsertPlugin(plugin: Plugin): void {
  db.prepare(
    `INSERT INTO plugins (id, name, version, tier, enabled, config)
     VALUES (?, ?, ?, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET
       name=excluded.name, version=excluded.version,
       tier=excluded.tier, enabled=excluded.enabled, config=excluded.config`
  ).run(plugin.id, plugin.name, plugin.version, plugin.tier, plugin.enabled ? 1 : 0, JSON.stringify(plugin.config));
}

export function getAlerts(acknowledged?: boolean): Alert[] {
  if (acknowledged !== undefined) return db.prepare("SELECT * FROM alerts WHERE acknowledged = ? ORDER BY created_at DESC").all(acknowledged ? 1 : 0) as Alert[];
  return db.prepare("SELECT * FROM alerts ORDER BY created_at DESC").all() as Alert[];
}

export function insertAlert(alert: Alert): void {
  db.prepare(
    `INSERT INTO alerts (id, event_id, rule_id, title, message, severity, acknowledged, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
  ).run(alert.id, alert.event_id, alert.rule_id, alert.title, alert.message, alert.severity, alert.acknowledged ? 1 : 0, alert.created_at);
}

export function getClips(device_id?: string): Clip[] {
  if (device_id) return db.prepare("SELECT * FROM clips WHERE device_id = ? ORDER BY timestamp DESC").all(device_id) as Clip[];
  return db.prepare("SELECT * FROM clips ORDER BY timestamp DESC").all() as Clip[];
}

export function getTelemetry(device_id: string, limit = 100): Telemetry[] {
  return db.prepare("SELECT * FROM telemetry WHERE device_id = ? ORDER BY timestamp DESC LIMIT ?").all(device_id, limit) as Telemetry[];
}

// ponytail: expose raw run for admin scripts
export function run(sql: string, ...params: unknown[]) {
  return db.prepare(sql).run(...params);
}

export function insertTelemetry(t: Telemetry): void {
  db.prepare(
    `INSERT INTO telemetry (device_id, timestamp, cpu_usage, gpu_usage, memory_usage, temperature, uptime)
     VALUES (?, ?, ?, ?, ?, ?, ?)`
  ).run(t.device_id, t.timestamp, t.cpu_usage, t.gpu_usage, t.memory_usage, t.temperature, t.uptime);
}

export function getAggregates(device_id: string, hours = 24): HourlyAggregate[] {
  return db.prepare(
    "SELECT * FROM hourly_aggregates WHERE device_id = ? ORDER BY hour DESC LIMIT ?"
  ).all(device_id, hours) as HourlyAggregate[];
}

export function upsertAggregate(agg: HourlyAggregate): void {
  db.prepare(
    `INSERT INTO hourly_aggregates (device_id, hour, vehicle_count, vehicle_counts, avg_speed, p50_speed, p95_speed, max_speed, event_counts)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(device_id, hour) DO UPDATE SET
       vehicle_count=excluded.vehicle_count, vehicle_counts=excluded.vehicle_counts,
       avg_speed=excluded.avg_speed, p50_speed=excluded.p50_speed,
       p95_speed=excluded.p95_speed, max_speed=excluded.max_speed,
       event_counts=excluded.event_counts`
  ).run(agg.device_id, agg.hour, agg.vehicle_count, JSON.stringify(agg.vehicle_counts),
    agg.avg_speed, agg.p50_speed, agg.p95_speed, agg.max_speed, JSON.stringify(agg.event_counts));
}

export function getTwinState(twin_path?: string): TwinState[] {
  if (twin_path) return db.prepare("SELECT * FROM twin_state WHERE twin_path = ? ORDER BY recorded_at DESC LIMIT 1").all(twin_path) as TwinState[];
  return db.prepare("SELECT * FROM twin_state ORDER BY twin_path").all() as TwinState[];
}

export function upsertTwinState(state: TwinState): void {
  db.prepare(
    `INSERT INTO twin_state (id, twin_path, state, confidence, recorded_at)
     VALUES (?, ?, ?, ?, ?)
     ON CONFLICT(twin_path) DO UPDATE SET
       state=excluded.state, confidence=excluded.confidence, recorded_at=excluded.recorded_at`
  ).run(state.id, state.twin_path, JSON.stringify(state.state), state.confidence, state.recorded_at);
}

export function getUserByEmail(email: string): { id: string; email: string; password_hash: string; role: string } | undefined {
  return db.prepare("SELECT * FROM users WHERE email = ?").get(email) as any;
}

export default db;
