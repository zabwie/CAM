import { useEffect, useRef, useState, useCallback } from "react";

interface WsMessage {
  type: "event" | "alert" | "device_status" | "health";
  payload: unknown;
}

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WsMessage | null>(null);
  const listenersRef = useRef<((msg: WsMessage) => void)[]>([]);

  useEffect(() => {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${location.host}/ws`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as WsMessage;
        setLastMessage(msg);
        listenersRef.current.forEach((fn) => fn(msg));
      } catch { /* ignore */ }
    };

    return () => ws.close();
  }, []);

  const subscribe = useCallback((filters: { event_types?: string[]; device_ids?: string[] }) => {
    wsRef.current?.send(JSON.stringify({ type: "subscribe", ...filters }));
  }, []);

  const addListener = useCallback((fn: (msg: WsMessage) => void) => {
    listenersRef.current.push(fn);
    return () => { listenersRef.current = listenersRef.current.filter((l) => l !== fn); };
  }, []);

  return { connected, lastMessage, subscribe, addListener };
}
