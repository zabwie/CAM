import { Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout.tsx";
import Dashboard from "./pages/Dashboard.tsx";
import Events from "./pages/Events.tsx";
import Devices from "./pages/Devices.tsx";
import Analytics from "./pages/Analytics.tsx";
import Calibrations from "./pages/Calibrations.tsx";
import Alerts from "./pages/Alerts.tsx";
import Plugins from "./pages/Plugins.tsx";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/events" element={<Events />} />
        <Route path="/devices" element={<Devices />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/calibrations" element={<Calibrations />} />
        <Route path="/alerts" element={<Alerts />} />
        <Route path="/plugins" element={<Plugins />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}
