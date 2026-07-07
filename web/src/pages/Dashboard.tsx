import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

export default function Dashboard() {
  const [stats, setStats] = useState({ devices: 0, eventsToday: 0, alerts: 0, avgSpeed: 0 });
  const [recent, setRecent] = useState<any[]>([]);

  useEffect(() => {
    fetch("/api/v1/events?limit=10").then((r) => r.json()).then(setRecent).catch(() => {});
    fetch("/api/v1/devices").then((r) => r.json()).then((d) => setStats((s) => ({ ...s, devices: d.length }))).catch(() => {});
    fetch("/api/v1/alerts").then((r) => r.json()).then((a) => setStats((s) => ({ ...s, alerts: a.length }))).catch(() => {});
    fetch("/api/v1/analytics/speed").then((r) => r.json()).then((a: any[]) => {
      if (a.length) setStats((s) => ({ ...s, avgSpeed: Math.round(a[a.length - 1].avg_speed), eventsToday: a.reduce((t: number, h: any) => t + h.vehicle_count, 0) }));
    }).catch(() => {});
  }, []);

  const chartData = recent.slice(0, 10).map((e: any) => ({
    name: (e.type || "?").split(".").pop(),
    speed: e.metadata?.speed_kmh || 0,
  }));

  return (
    <div>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 24 }}>Dashboard</h1>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 32 }}>
        {[
          { label: "Devices", value: stats.devices, color: "#38bdf8" },
          { label: "Events Today", value: stats.eventsToday, color: "#a78bfa" },
          { label: "Active Alerts", value: stats.alerts, color: "#f87171" },
          { label: "Avg Speed", value: `${stats.avgSpeed} km/h`, color: "#34d399" },
        ].map((c) => (
          <div key={c.label} style={{ background: "#1e293b", borderRadius: 12, padding: "20px" }}>
            <div style={{ fontSize: 12, color: "#94a3b8", textTransform: "uppercase", letterSpacing: 1 }}>{c.label}</div>
            <div style={{ fontSize: 32, fontWeight: 700, color: c.color, marginTop: 8 }}>{c.value}</div>
          </div>
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 24 }}>
        <div style={{ background: "#1e293b", borderRadius: 12, padding: 20 }}>
          <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 16 }}>Recent Speeds</h2>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={chartData}>
              <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8 }} />
              <Bar dataKey="speed" fill="#38bdf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div style={{ background: "#1e293b", borderRadius: 12, padding: 20 }}>
          <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 16 }}>Recent Events</h2>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {recent.slice(0, 6).map((e: any) => (
              <div key={e.id} style={{ fontSize: 13, padding: "8px 0", borderBottom: "1px solid #334155" }}>
                <span style={{ color: "#38bdf8" }}>{e.type}</span>
                <span style={{ color: "#64748b", marginLeft: 8 }}>{e.device_id?.slice(0, 8)}</span>
                <div style={{ color: "#94a3b8", fontSize: 11 }}>
                  {e.metadata?.speed_kmh ? `${e.metadata.speed_kmh} km/h` : ""}
                  {e.metadata?.vehicle_class ? ` · ${e.metadata.vehicle_class}` : ""}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
