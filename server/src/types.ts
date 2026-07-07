import { z } from "zod";

// ── Event ──
export const EventSchema = z.object({
  id: z.string().uuid(),
  ts: z.number(), // nanosecond precision
  device_id: z.string(),
  type: z.string(), // namespaced: "traffic.speed", "safety.wrong_way"
  plugin_id: z.string(),
  severity: z.number().int().min(0).max(2), // 0=info, 1=warning, 2=critical
  confidence: z.number().min(0).max(1),
  source: z.number().int().min(0).max(2), // 0=auto, 1=verified, 2=manual
  metadata: z.record(z.unknown()).default({}),
  clip_id: z.string().nullable().optional(),
  twin_path: z.string().nullable().optional(),
});
export type Event = z.infer<typeof EventSchema>;

// ── Vehicle ──
export interface Vehicle {
  id: string;
  event_id: string;
  track_id: number;
  vehicle_class: string;
  speed: number;
  direction: number;
  lane: number;
  trajectory: { x: number; y: number; ts: number }[];
  features: Record<string, unknown>;
}

// ── Device ──
export interface Device {
  id: string;
  name: string;
  location: { lat: number; lng: number };
  hardware_model: string;
  os_version: string;
  firmware: string;
  status: "provisioning" | "online" | "offline" | "degraded" | "error";
  last_seen: number;
  config: Record<string, unknown>;
  capabilities: Record<string, unknown>;
  health: Record<string, unknown>;
}

// ── Calibration ──
export interface Calibration {
  id: string;
  device_id: string;
  version: number;
  homography: number[][];
  lanes: Record<string, unknown>;
  zones: Record<string, unknown>;
  confidence: number;
  created_by: string;
  created_at: string;
  active: boolean;
}

// ── Plugin ──
export interface Plugin {
  id: string;
  name: string;
  version: string;
  tier: string;
  enabled: boolean;
  config: Record<string, unknown>;
}

// ── Alert ──
export interface Alert {
  id: string;
  event_id: string;
  rule_id: string;
  title: string;
  message: string;
  severity: "info" | "warning" | "critical";
  acknowledged: boolean;
  created_at: number;
}

// ── Clip ──
export interface Clip {
  id: string;
  device_id: string;
  timestamp: number;
  duration: number;
  size_bytes: number;
  storage_path: string;
  status: "uploading" | "ready" | "archived" | "deleted";
}

// ── Telemetry ──
export interface Telemetry {
  device_id: string;
  timestamp: number;
  cpu_usage: number;
  gpu_usage: number;
  memory_usage: number;
  temperature: number;
  uptime: number;
}

// ── Aggregate ──
export interface HourlyAggregate {
  device_id: string;
  hour: string;
  vehicle_count: number;
  vehicle_counts: Record<string, number>;
  avg_speed: number;
  p50_speed: number;
  p95_speed: number;
  max_speed: number;
  event_counts: Record<string, number>;
}

// ── Digital Twin State ──
export interface TwinState {
  id: string;
  twin_path: string;
  state: Record<string, unknown>;
  confidence: number;
  recorded_at: number;
}

// ── WebSocket Messages ──
export type WsMessage =
  | { type: "subscribe"; event_types?: string[]; device_ids?: string[] }
  | { type: "unsubscribe"; event_types?: string[]; device_ids?: string[] }
  | { type: "event"; payload: Event }
  | { type: "alert"; payload: Alert }
  | { type: "device_status"; payload: Device }
  | { type: "health"; payload: Telemetry };
