import { EventEmitter } from "events";
import type { Event } from "./types.js";

// ponytail: in-process EventEmitter for v1. Swap for Redis/NATS when multi-process needed.
const emitter = new EventEmitter();
emitter.setMaxListeners(100);

export interface Subscription {
  remove: () => void;
}

export function publish(event: Event): void {
  emitter.emit("event", event);
  emitter.emit(`event:${event.type}`, event);
  emitter.emit(`device:${event.device_id}`, event);
}

export function subscribe(handler: (event: Event) => void): Subscription {
  emitter.on("event", handler);
  return { remove: () => emitter.off("event", handler) };
}

export function subscribeType(
  eventType: string,
  handler: (event: Event) => void
): Subscription {
  emitter.on(`event:${eventType}`, handler);
  return { remove: () => emitter.off(`event:${eventType}`, handler) };
}

export function subscribeDevice(
  deviceId: string,
  handler: (event: Event) => void
): Subscription {
  emitter.on(`device:${deviceId}`, handler);
  return { remove: () => emitter.off(`device:${deviceId}`, handler) };
}
