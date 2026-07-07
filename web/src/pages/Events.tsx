import { useEffect, useState } from "react";

export default function Events() {
  const [events, setEvents] = useState<any[]>([]);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    fetch(`/api/v1/events${filter ? `?type=${filter}` : ""}`)
      .then((r) => r.json()).then(setEvents).catch(() => {});
  }, [filter]);

  const types = [...new Set(events.map((e: any) => e.type))];

  return (
    <div>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 24 }}>Events</h1>
      <div style={{ marginBottom: 16, display: "flex", gap: 8 }}>
        <button onClick={() => setFilter("")} style={btnStyle(!filter)}>All</button>
        {types.map((t) => (
          <button key={t as string} onClick={() => setFilter(t as string)} style={btnStyle(filter === t)}>
            {t as string}
          </button>
        ))}
      </div>
      <div style={{ background: "#1e293b", borderRadius: 12, overflow: "hidden" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ color: "#94a3b8", borderBottom: "1px solid #334155" }}>
              {["Time", "Type", "Device", "Speed", "Class", "Confidence"].map((h) => (
                <th key={h} style={{ textAlign: "left", padding: "12px 16px", fontWeight: 500 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {events.map((e: any) => (
              <tr key={e.id} style={{ borderBottom: "1px solid #1e293b" }}>
                <td style={{ padding: "10px 16px", color: "#94a3b8" }}>{new Date(e.ts / 1_000_000).toLocaleTimeString()}</td>
                <td style={{ padding: "10px 16px" }}><span style={{ color: "#38bdf8" }}>{e.type}</span></td>
                <td style={{ padding: "10px 16px", color: "#a78bfa" }}>{e.device_id?.slice(0, 12)}</td>
                <td style={{ padding: "10px 16px" }}>{e.metadata?.speed_kmh ?? "—"}</td>
                <td style={{ padding: "10px 16px" }}>{e.metadata?.vehicle_class ?? "—"}</td>
                <td style={{ padding: "10px 16px" }}>{(e.confidence * 100).toFixed(0)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function btnStyle(active: boolean): React.CSSProperties {
  return {
    padding: "6px 14px", borderRadius: 8, border: "none", cursor: "pointer",
    fontSize: 13, fontWeight: 500,
    background: active ? "#38bdf8" : "#1e293b",
    color: active ? "#0f172a" : "#94a3b8",
  };
}
