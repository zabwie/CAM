import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line } from "recharts";

export default function Analytics() {
  const [data, setData] = useState<any[]>([]);

  useEffect(() => {
    fetch("/api/v1/analytics/speed")
      .then((r) => r.json()).then((d: any[]) => {
        setData(d.slice(-48).map((h: any) => ({
          hour: h.hour?.slice(11, 16) || h.hour,
          avg_speed: Math.round(h.avg_speed),
          p95_speed: Math.round(h.p95_speed),
          max_speed: Math.round(h.max_speed),
          count: h.vehicle_count,
        })));
      }).catch(() => {});
  }, []);

  return (
    <div>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 24 }}>Analytics</h1>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        <div style={{ background: "#1e293b", borderRadius: 12, padding: 20 }}>
          <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 16 }}>Average Speed (km/h)</h2>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={data}>
              <XAxis dataKey="hour" tick={{ fill: "#94a3b8", fontSize: 10 }} />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} />
              <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8 }} />
              <Line type="monotone" dataKey="avg_speed" stroke="#38bdf8" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="p95_speed" stroke="#f87171" strokeWidth={1} dot={false} strokeDasharray="4 4" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div style={{ background: "#1e293b", borderRadius: 12, padding: 20 }}>
          <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 16 }}>Vehicle Count</h2>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={data}>
              <XAxis dataKey="hour" tick={{ fill: "#94a3b8", fontSize: 10 }} />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} />
              <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8 }} />
              <Bar dataKey="count" fill="#a78bfa" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
