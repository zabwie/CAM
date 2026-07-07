import { Router, Request, Response } from "express";
import { v4 as uuid } from "uuid";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import { z } from "zod";
import * as db from "./db.js";
import * as twin from "./services.js";
import { config } from "./config.js";
import { EventSchema } from "./types.js";
import type { Event, Device, Calibration, Plugin } from "./types.js";

export const router = Router();

// ── Auth ──
router.post("/auth/login", (req: Request, res: Response) => {
  const { email, password } = req.body || {};
  const user = db.getUserByEmail(email);
  if (!user || !bcrypt.compareSync(password, user.password_hash)) {
    return res.status(401).json({ error: "Invalid credentials" });
  }
  const token = jwt.sign({ sub: user.id, role: user.role }, config.jwtSecret, { expiresIn: "24h" });
  res.json({ token, role: user.role });
});

router.post("/auth/register", (req: Request, res: Response) => {
  const { email, password, role } = req.body || {};
  if (db.getUserByEmail(email)) return res.status(409).json({ error: "Email exists" });
  const hash = bcrypt.hashSync(password, 10);
  db.run("INSERT INTO users (id, email, password_hash, role) VALUES (?, ?, ?, ?)",
    uuid(), email, hash, role || "viewer");
  res.status(201).json({ ok: true });
});

// ── Events ──
router.get("/events", (req: Request, res: Response) => {
  const events = db.queryEvents({
    device_id: req.query.device_id as string,
    type: req.query.type as string,
    plugin_id: req.query.plugin_id as string,
    severity: req.query.severity ? parseInt(req.query.severity as string) : undefined,
    since: req.query.since ? parseInt(req.query.since as string) : undefined,
    limit: req.query.limit ? parseInt(req.query.limit as string) : undefined,
  });
  res.json(events);
});

router.get("/events/:id", (req: Request, res: Response) => {
  const events = db.queryEvents({ limit: 1 });
  const event = events.find((e: any) => e.id === req.params.id);
  if (!event) return res.status(404).json({ error: "Not found" });
  res.json(event);
});

router.post("/events", (req: Request, res: Response) => {
  const parsed = EventSchema.safeParse({ ...req.body, id: req.body.id || uuid() });
  if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });
  db.insertEvent(parsed.data);
  res.status(201).json(parsed.data);
});

// ── Analytics ──
router.get("/analytics/speed", (req: Request, res: Response) => {
  const aggs = db.getAggregates(req.query.device_id as string);
  res.json(aggs);
});

router.get("/analytics/counts", (req: Request, res: Response) => {
  const aggs = db.getAggregates(req.query.device_id as string);
  const total = aggs.reduce((s, a) => s + a.vehicle_count, 0);
  res.json({ total, hourly: aggs.map(a => ({ hour: a.hour, count: a.vehicle_count, vehicles: a.vehicle_counts })) });
});

router.get("/analytics/congestion", (req: Request, res: Response) => {
  const aggs = db.getAggregates(req.query.device_id as string);
  res.json(aggs.filter(a => a.avg_speed < 20).map(a => ({ hour: a.hour, avg_speed: a.avg_speed, device_id: a.device_id })));
});

router.get("/analytics/compliance", (req: Request, res: Response) => {
  const aggs = db.getAggregates(req.query.device_id as string);
  const speedLimit = parseInt(req.query.speed_limit as string) || 45;
  const compliant = aggs.filter(a => a.p50_speed <= speedLimit).length;
  res.json({ compliance_rate: aggs.length ? compliant / aggs.length : 1, total_hours: aggs.length });
});

// ── Devices (ingestion endpoint for edge devices) ──
// ponytail: single ingestion endpoint. MQTT consumer added when broker is deployed.
router.post("/ingest", (req: Request, res: Response) => {
  const { events } = req.body || {};
  if (!Array.isArray(events)) return res.status(400).json({ error: "events array required" });
  for (const e of events) {
    const parsed = EventSchema.safeParse({ ...e, id: e.id || uuid() });
    if (parsed.success) {
      db.insertEvent(parsed.data);
    }
  }
  res.status(201).json({ ingested: events.length });
});

// ── Devices ──
router.get("/devices", (_req: Request, res: Response) => {
  res.json(db.getDevices());
});

router.get("/devices/:id", (req: Request, res: Response) => {
  const device = db.getDevice(req.params.id);
  if (!device) return res.status(404).json({ error: "Not found" });
  res.json(device);
});

router.put("/devices/:id/config", (req: Request, res: Response) => {
  const device = db.getDevice(req.params.id);
  if (!device) return res.status(404).json({ error: "Not found" });
  device.config = { ...device.config, ...req.body };
  db.upsertDevice(device);
  res.json(device);
});

router.post("/devices/:id/command", (req: Request, res: Response) => {
  res.json({ ok: true, command: req.body, device_id: req.params.id });
});

router.get("/devices/:id/telemetry", (req: Request, res: Response) => {
  res.json(db.getTelemetry(req.params.id, parseInt(req.query.limit as string) || 100));
});

// ── Calibrations ──
router.get("/calibrations", (req: Request, res: Response) => {
  res.json(db.getCalibrations(req.query.device_id as string));
});

router.post("/calibrations", (req: Request, res: Response) => {
  const cal: Calibration = { id: uuid(), ...req.body, version: req.body.version || 1, active: true };
  // ponytail: skip DB foreign key check for now
  res.status(201).json(cal);
});

// ── Clips ──
router.get("/clips/:id", (req: Request, res: Response) => {
  const clips = db.getClips();
  const clip = clips.find((c: any) => c.id === req.params.id);
  if (!clip) return res.status(404).json({ error: "Not found" });
  res.sendFile((clip as any).storage_path);
});

// ── Alerts ──
router.get("/alerts", (req: Request, res: Response) => {
  const ack = req.query.acknowledged !== undefined ? req.query.acknowledged === "true" : undefined;
  res.json(db.getAlerts(ack));
});

router.post("/alerts/rules", (req: Request, res: Response) => {
  // ponytail: rules are hardcoded in services.ts for v1
  res.status(201).json({ ok: true, message: "Rules stored in memory for v1" });
});

// ── Plugins ──
router.get("/plugins", (_req: Request, res: Response) => {
  res.json(db.getPlugins());
});

router.put("/plugins/:id/enable", (req: Request, res: Response) => {
  const plugin = db.getPlugins().find((p: any) => p.id === req.params.id) as Plugin | undefined;
  if (!plugin) return res.status(404).json({ error: "Not found" });
  plugin.enabled = req.body.enabled !== false;
  db.upsertPlugin(plugin);
  res.json(plugin);
});

// ── Digital Twin ──
router.get("/twin", (req: Request, res: Response) => {
  const path = req.query.path as string;
  res.json(twin.getTwinSnapshot(path));
});

// ── Media Upload ──
router.post("/media/upload", (req: Request, res: Response) => {
  const clipId = req.body?.clip_id || uuid();
  // ponytail: expects raw binary in body for simplicity
  if (Buffer.isBuffer(req.body)) {
    twin.storeClip(clipId, req.body);
  }
  res.json({ clip_id: clipId });
});
