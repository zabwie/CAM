import express from "express";
import cors from "cors";
import helmet from "helmet";
import jwt from "jsonwebtoken";
import http from "http";
import { WebSocketServer } from "ws";
import bcrypt from "bcryptjs";
import { v4 as uuid } from "uuid";
import { config } from "./config.js";
import { initSchema, upsertPlugin, getUserByEmail } from "./db.js";
import { router } from "./api.js";
import {
  startPersistence,
  startAnalytics,
  startAlerts,
  startWsPublisher,
  startDigitalTwin,
  startSyncAgent,
} from "./services.js";

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: "/ws" });

// ── Middleware ──
app.use(helmet());
app.use(cors({ origin: "*" }));
app.use(express.json({ limit: "50mb" }));
app.use(express.raw({ type: "application/octet-stream", limit: "100mb" }));

app.use("/api/v1", (req, res, next) => {
  if (req.path.startsWith("/auth/") || req.path === "/ingest") return next();
  const auth = req.headers.authorization;
  if (!auth || !auth.startsWith("Bearer ")) {
    if (config.nodeEnv === "development") return next();
    return res.status(401).json({ error: "Missing token" });
  }
  try {
    (req as any).user = jwt.verify(auth.slice(7), config.jwtSecret);
    next();
  } catch { res.status(401).json({ error: "Invalid token" }); }
});

app.use("/api/v1", router);
app.get("/health", (_req, res) => res.json({ status: "ok", uptime: process.uptime() }));

// ── Init ──
initSchema();

upsertPlugin({ id: "speed", name: "Speed Detection", version: "0.1.0", tier: "starter", enabled: true, config: {} });
upsertPlugin({ id: "wrong_way", name: "Wrong Way Detection", version: "0.1.0", tier: "professional", enabled: true, config: {} });
upsertPlugin({ id: "congestion", name: "Congestion Analysis", version: "0.1.0", tier: "professional", enabled: true, config: {} });
upsertPlugin({ id: "stopped_vehicle", name: "Stopped Vehicle Detection", version: "0.1.0", tier: "enterprise", enabled: true, config: {} });

if (!getUserByEmail("admin@cam.local")) {
  const { run } = await import("./db.js");
  run("INSERT OR IGNORE INTO users (id, email, password_hash, role) VALUES (?, ?, ?, ?)",
    uuid(), "admin@cam.local", bcrypt.hashSync("admin", 10), "admin");
}

startPersistence();
startAnalytics();
startAlerts();
startWsPublisher(wss);
startDigitalTwin();
startSyncAgent();

server.listen(config.port, () => {
  console.log(`[cam-server] listening on :${config.port}`);
  console.log(`[cam-server] WebSocket at ws://localhost:${config.port}/ws`);
  console.log(`[cam-server] environment: ${config.nodeEnv}`);
});
