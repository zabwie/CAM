import { useEffect, useState } from "react";

export default function Alerts() {
  const [alerts, setAlerts] = useState<any[]>([]);

  useEffect(() => {
    fetch("/api/v1/alerts").then((r) => r.json()).then(setAlerts).catch(() => {});
  }, []);

  const sevColor: Record<string, string> = { info: "#94a3b8", warning: "#fbbf24", critical: "#ef4444" };

  return (
    <div>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 24 }}>Alerts</h1>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {alerts.map((a: any) => (
          <div key={a.id} style={{
            background: "#1e293b", borderRadius: 10, padding: "14px 18px",
            display: "flex", alignItems: "center", gap: 12,
            opacity: a.acknowledged ? 0.5 : 1,
          }}>
            <div style={{
              width: 8, height: 8, borderRadius: "50%",
              background: sevColor[a.severity] || "#94a3b8",
              flexShrink: 0,
            }} />
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 600, fontSize: 14 }}>{a.title}</div>
              <div style={{ fontSize: 12, color: "#94a3b8" }}>{a.message}</div>
            </div>
            <div style={{ fontSize: 11, color: "#64748b" }}>
              {new Date(a.created_at).toLocaleString()}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
