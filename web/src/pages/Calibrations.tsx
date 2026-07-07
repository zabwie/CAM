import { useEffect, useState } from "react";

export default function Calibrations() {
  const [cals, setCals] = useState<any[]>([]);
  useEffect(() => {
    fetch("/api/v1/calibrations").then((r) => r.json()).then(setCals).catch(() => {});
  }, []);

  return (
    <div>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 24 }}>Calibrations</h1>
      <div style={{ background: "#1e293b", borderRadius: 12, overflow: "hidden" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ color: "#94a3b8", borderBottom: "1px solid #334155" }}>
              {["Device", "Version", "Confidence", "Created By", "Active", "Created At"].map((h) => (
                <th key={h} style={{ textAlign: "left", padding: "12px 16px", fontWeight: 500 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {cals.map((c: any) => (
              <tr key={c.id} style={{ borderBottom: "1px solid #1e293b" }}>
                <td style={{ padding: "10px 16px", color: "#a78bfa" }}>{c.device_id?.slice(0, 12)}</td>
                <td style={{ padding: "10px 16px" }}>v{c.version}</td>
                <td style={{ padding: "10px 16px" }}>{(c.confidence * 100).toFixed(0)}%</td>
                <td style={{ padding: "10px 16px", color: "#94a3b8" }}>{c.created_by}</td>
                <td style={{ padding: "10px 16px" }}>
                  <span style={{ color: c.active ? "#34d399" : "#f87171" }}>{c.active ? "Yes" : "No"}</span>
                </td>
                <td style={{ padding: "10px 16px", color: "#94a3b8" }}>{c.created_at}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
