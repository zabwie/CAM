import { useEffect, useState } from "react";

export default function Devices() {
  const [devices, setDevices] = useState<any[]>([]);

  useEffect(() => {
    fetch("/api/v1/devices").then((r) => r.json()).then(setDevices).catch(() => {});
  }, []);

  const statusColor: Record<string, string> = { online: "#34d399", offline: "#f87171", degraded: "#fbbf24", error: "#ef4444", provisioning: "#94a3b8" };

  return (
    <div>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 24 }}>Devices</h1>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: 16 }}>
        {devices.map((d: any) => (
          <div key={d.id} style={{ background: "#1e293b", borderRadius: 12, padding: 20 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <span style={{ fontWeight: 600 }}>{d.name || d.id.slice(0, 12)}</span>
              <span style={{
                padding: "3px 10px", borderRadius: 99, fontSize: 11, fontWeight: 600,
                background: statusColor[d.status] + "22", color: statusColor[d.status],
              }}>
                {d.status}
              </span>
            </div>
            <div style={{ fontSize: 12, color: "#94a3b8", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 16px" }}>
              <span>Model: {d.hardware_model || "—"}</span>
              <span>Firmware: {d.firmware || "—"}</span>
              <span>Last seen: {d.last_seen ? new Date(d.last_seen).toLocaleString() : "—"}</span>
              <span>OS: {d.os_version || "—"}</span>
            </div>
          </div>
        ))}
      </div>
      {devices.length === 0 && (
        <div style={{ color: "#64748b", textAlign: "center", padding: 60 }}>
          No devices registered. Devices appear when they connect to the server.
        </div>
      )}
    </div>
  );
}
