import { ReactNode } from "react";
import Sidebar from "./Sidebar.tsx";

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div style={{ display: "flex", height: "100vh", background: "#0f172a", color: "#e2e8f0" }}>
      <Sidebar />
      <main style={{ flex: 1, overflow: "auto", padding: "24px" }}>
        {children}
      </main>
    </div>
  );
}
