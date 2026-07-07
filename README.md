# Cam

Intelligent traffic management platform. Turns existing cameras into real-time traffic sensors.

## What it is

Cam is an **operating system for traffic safety**, not a speed camera company. Speed detection is the wedge feature; the platform does traffic analytics, public safety, school zone monitoring, and city planning.

## Architecture at a Glance

```
Camera → Edge Device (C++ inference) → Local Server (TS) → Cloud (TS)
```

- **Edge**: Runs AI inference on camera feeds in real time — detects vehicles, measures speed, generates events. Works offline, syncs when connected.
- **Local Server**: Per-municipality backend — event DB, dashboard, video archive. Containerized, runs on-prem or in the customer's cloud.
- **Cloud**: Fleet management, model updates, aggregated multi-city analytics, licensing, support.

Same codebase, three deployment tiers: SaaS (small towns), Hybrid (medium), On-prem (enterprise).

## Product Tiers

| Tier         | Target         | Hosting      |
|-------------|----------------|--------------|
| Starter     | Small towns    | Full SaaS    |
| Professional| Medium cities  | Hybrid       |
| Enterprise  | Large cities   | On-prem/private cloud |

## Repo Structure

```
cam/
├── edge/              # C++ inference pipeline (runs on device)
├── server/            # TypeScript local/cloud backend
├── web/               # Dashboard UI
├── deploy/            # Docker Compose, Helm charts
└── docs/              # Architecture, API specs
```

## Language Choices

- **Edge pipeline**: C++ — zero-cost abstraction, direct GPU access, deterministic latency.
- **Server/API/Cloud**: TypeScript (Node/Deno) — fast iteration, rich ecosystem, same language for frontend and backend.
- **Infrastructure**: Docker containers everywhere — same image runs in your cloud or a municipality's server room.
