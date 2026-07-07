import { useEffect, useState } from "react";

export default function Plugins() {
  const [plugins, setPlugins] = useState<any[]>([]);

  useEffect(() => {
    fetch("/api/v1/plugins").then((r) => r.json()).then(setPlugins).catch(() => {});
  }, []);

  const tierColor: Record<string, string> = { starter: "#94a3b8", professional: "#38bdf8", enterprise: "#a78bfa" };

  return (
    <div>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 24 }}>Plugins</h1>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 16 }}>
        {plugins.map((p: any) => (
          <div key={p.id} style={{ background: "#1e293b", borderRadius: 12, padding: 20 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <span style={{ fontWeight: 600 }}>{p.name}</span>
              <span style={{
                padding: "2px 8px", borderRadius: 99, fontSize: 10, fontWeight: 600, textTransform: "uppercase",
                color: tierColor[p.tier] || "#94a3b8",
                background: (tierColor[p.tier] || "#94a3b8") + "22",
              }}>
                {p.tier}
              </span>
            </div>
            <div style={{ fontSize: 12, color: "#94a3b8" }}>
              v{p.version} · {p.enabled ? "Enabled" : "Disabled"}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
