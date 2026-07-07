import { NavLink } from "react-router-dom";

const links = [
  { to: "/", label: "Dashboard", icon: "◉" },
  { to: "/events", label: "Events", icon: "⚡" },
  { to: "/devices", label: "Devices", icon: "📷" },
  { to: "/analytics", label: "Analytics", icon: "📊" },
  { to: "/calibrations", label: "Calibrations", icon: "⚙" },
  { to: "/alerts", label: "Alerts", icon: "🔔" },
  { to: "/plugins", label: "Plugins", icon: "🧩" },
];

export default function Sidebar() {
  return (
    <nav style={{ width: 220, background: "#1e293b", padding: "16px 0", display: "flex", flexDirection: "column", gap: "2px" }}>
      <div style={{ padding: "12px 20px", fontSize: 20, fontWeight: 700, color: "#38bdf8", marginBottom: 16 }}>
        Cam
      </div>
      {links.map((l) => (
        <NavLink
          key={l.to}
          to={l.to}
          end={l.to === "/"}
          style={({ isActive }) => ({
            display: "flex", alignItems: "center", gap: 10,
            padding: "10px 20px", textDecoration: "none",
            color: isActive ? "#38bdf8" : "#94a3b8",
            background: isActive ? "#0f172a" : "transparent",
            borderRight: isActive ? "3px solid #38bdf8" : "3px solid transparent",
            fontSize: 14, fontWeight: isActive ? 600 : 400,
          })}
        >
          <span>{l.icon}</span> {l.label}
        </NavLink>
      ))}
    </nav>
  );
}
