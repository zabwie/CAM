// ponytail: all services in one file. Split into separate modules when any service needs independent deployment.
import { v4 as uuid } from "uuid";
import * as bus from "./event-bus.js";
import * as db from "./db.js";
import type { Event, Telemetry, Alert, HourlyAggregate, TwinState } from "./types.js";

// ── Ingestion Service ──
// ponytail: HTTP endpoint for now. MQTT consumer added when MQTT broker is deployed.
export function startIngestion(): void {
  bus.subscribeType("ingestion.event", (event: Event) => {
    bus.publish(event);
  });
  console.log("[ingestion] started (HTTP mode)");
}

// ── Persistence Service ──
// Writes every event + vehicle to the database.
export function startPersistence(): void {
  bus.subscribe((event) => {
    try {
      db.insertEvent(event);
    } catch (err) {
      console.error("[persistence] failed to write event:", err);
    }
  });
  console.log("[persistence] started");
}

// ── Analytics Service ──
export function startAnalytics(): void {
  const aggBuffer = new Map<string, Record<string, unknown[]>>();

  bus.subscribe((event) => {
    try {
      const hour = new Date(event.ts / 1_000_000).toISOString().slice(0, 13) + ":00";
      const key = `${event.device_id}:${hour}`;

      if (!aggBuffer.has(key)) aggBuffer.set(key, { speeds: [], classes: [], eventTypes: [] });
      const bucket = aggBuffer.get(key)!;

      if (event.metadata?.speed_kmh != null) (bucket.speeds as number[]).push(event.metadata.speed_kmh as number);
      const vclass = (event.metadata?.vehicle_class as string) || "unknown";
      (bucket.classes as string[]).push(vclass);
      (bucket.eventTypes as string[]).push(event.type);
    } catch { /* skip malformed */ }
  });

  // Flush aggregates every 60s
  setInterval(() => {
    const now = new Date();
    const currentHour = now.toISOString().slice(0, 13) + ":00";
    for (const [key, bucket] of aggBuffer) {
      const [deviceId, hour] = key.split(":");
      if (hour >= currentHour) continue;

      const speeds = bucket.speeds as number[];
      const classes = bucket.classes as string[];
      const eventTypes = bucket.eventTypes as string[];

      const sorted = [...speeds].sort((a, b) => a - b);
      const agg: HourlyAggregate = {
        device_id: deviceId,
        hour,
        vehicle_count: classes.length,
        vehicle_counts: Object.fromEntries(
          [...new Set(classes)].map((c) => [c, classes.filter((x) => x === c).length])
        ),
        avg_speed: speeds.length ? speeds.reduce((a, b) => a + b, 0) / speeds.length : 0,
        p50_speed: sorted[Math.floor(sorted.length * 0.5)] ?? 0,
        p95_speed: sorted[Math.floor(sorted.length * 0.95)] ?? 0,
        max_speed: sorted[sorted.length - 1] ?? 0,
        event_counts: Object.fromEntries(
          [...new Set(eventTypes)].map((t) => [t, eventTypes.filter((x) => x === t).length])
        ),
      };
      db.upsertAggregate(agg);
      aggBuffer.delete(key);
    }
  }, 60_000).unref();

  console.log("[analytics] started");
}

// ── Alert Service ──
// Evaluates rules against events.
export function startAlerts(): void {
  // ponytail: hardcoded rules. Load from DB when rule management UI exists.
  const rules = [
    { id: "speed-over-limit", type: "traffic.speed", field: "speed_kmh", op: "gt", value: 80, severity: "warning" as const, title: "Speeding detected" },
    { id: "critical-speed", type: "traffic.speed", field: "speed_kmh", op: "gt", value: 120, severity: "critical" as const, title: "Critical speeding" },
    { id: "wrong-way", type: "safety.wrong_way", field: null, op: "exists", value: null, severity: "critical" as const, title: "Wrong-way driver" },
  ];

  bus.subscribe((event) => {
    for (const rule of rules) {
      if (event.type !== rule.type) continue;
      let triggered = false;
      if (rule.op === "gt") {
        triggered = (event.metadata?.[rule.field!] as number) > (rule.value as number);
      } else if (rule.op === "exists") {
        triggered = true;
      }
      if (!triggered) continue;

      const alert: Alert = {
        id: uuid(),
        event_id: event.id,
        rule_id: rule.id,
        title: rule.title,
        message: `${rule.title} on ${event.device_id}: ${JSON.stringify(event.metadata)}`,
        severity: rule.severity,
        acknowledged: false,
        created_at: Date.now(),
      };
      db.insertAlert(alert);
      bus.publish({ ...event, type: "alert", plugin_id: "alerts", metadata: alert as any } as any);
    }
  });
  console.log("[alerts] started");
}

// ── WebSocket Publisher ──
import type { WebSocketServer, WebSocket } from "ws";

interface WsClient {
  ws: WebSocket;
  filters: { event_types?: Set<string>; device_ids?: Set<string> };
}

const clients: WsClient[] = [];

export function startWsPublisher(wss: WebSocketServer): void {
  wss.on("connection", (ws) => {
    const client: WsClient = { ws, filters: {} };
    clients.push(client);

    ws.on("message", (data) => {
      try {
        const msg = JSON.parse(data.toString());
        if (msg.type === "subscribe") {
          if (msg.event_types) client.filters.event_types = new Set(msg.event_types);
          if (msg.device_ids) client.filters.device_ids = new Set(msg.device_ids);
        }
      } catch { /* ignore */ }
    });

    ws.on("close", () => {
      const idx = clients.indexOf(client);
      if (idx >= 0) clients.splice(idx, 1);
    });
  });

  bus.subscribe((event) => {
    const payload = JSON.stringify({ type: "event", payload: event });
    for (const client of clients) {
      if (client.ws.readyState !== 1) continue;
      if (client.filters.event_types && !client.filters.event_types.has(event.type)) continue;
      if (client.filters.device_ids && !client.filters.device_ids.has(event.device_id)) continue;
      client.ws.send(payload);
    }
  });

  // Also subscribe to alerts
  bus.subscribe((event) => {
    if (event.type !== "alert") return;
    const payload = JSON.stringify({ type: "alert", payload: event.metadata });
    for (const client of clients) {
      if (client.ws.readyState === 1) client.ws.send(payload);
    }
  });

  console.log("[ws-publisher] started");
}

// ── Digital Twin Service ──
// Maintains in-memory state graph, persists snapshots.
const twinGraph = new Map<string, Record<string, unknown>>();

export function startDigitalTwin(): void {
  bus.subscribe((event) => {
    if (!event.twin_path) return;
    const path = event.twin_path;
    const current = twinGraph.get(path) || {};
    const update: Record<string, unknown> = {
      ...current,
      last_event: event.type,
      last_event_at: event.ts,
      last_severity: event.severity,
    };
    if (event.metadata?.speed_kmh != null) update.current_speed = event.metadata.speed_kmh;
    if (event.metadata?.vehicle_class != null) update.last_vehicle_class = event.metadata.vehicle_class;
    if (event.metadata?.lane != null) update.last_lane = event.metadata.lane;

    twinGraph.set(path, update);

    // Persist snapshot every 10 events
    const state: TwinState = {
      id: uuid(),
      twin_path: path,
      state: update,
      confidence: event.confidence,
      recorded_at: Date.now(),
    };
    db.upsertTwinState(state);
  });
  console.log("[digital-twin] started");
}

export function getTwinSnapshot(path?: string): Record<string, unknown> | Record<string, unknown>[] {
  if (path) return twinGraph.get(path) || {};
  return Object.fromEntries(twinGraph);
}

// ── Sync Agent ──
// ponytail: HTTP POST batching. Swap for NATS/ Kafka when cross-region needed.
export function startSyncAgent(): void {
  const cloudUrl = process.env.CLOUD_API_URL;
  const cloudKey = process.env.CLOUD_API_KEY;
  if (!cloudUrl || !cloudKey) {
    console.log("[sync-agent] disabled (CLOUD_API_URL not set)");
    return;
  }

  const buffer: Event[] = [];
  bus.subscribe((event) => {
    buffer.push(event);
  });

  setInterval(async () => {
    if (buffer.length === 0) return;
    const batch = buffer.splice(0, 100);
    try {
      const res = await fetch(`${cloudUrl}/api/v1/ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Key": cloudKey },
        body: JSON.stringify({ events: batch, tenant_id: process.env.TENANT_ID }),
      });
      if (!res.ok) console.warn("[sync-agent] batch upload failed:", res.status);
    } catch (err) {
      console.warn("[sync-agent] upload error:", err);
      buffer.unshift(...batch); // re-queue on failure
    }
  }, 30_000).unref();

  console.log("[sync-agent] started");
}

// ── Media Service ──
// ponytail: local filesystem storage. MinIO integration when multi-server needed.
import fs from "fs";
import path from "path";

const MEDIA_DIR = process.env.MEDIA_DIR || "./media";
if (!fs.existsSync(MEDIA_DIR)) fs.mkdirSync(MEDIA_DIR, { recursive: true });

export function getClipPath(clipId: string): string {
  return path.join(MEDIA_DIR, `${clipId}.mp4`);
}

export function storeClip(clipId: string, data: Buffer): string {
  const filePath = getClipPath(clipId);
  fs.writeFileSync(filePath, data);
  return filePath;
}
