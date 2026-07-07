#ifndef CAM_PLUGIN_H
#define CAM_PLUGIN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CAM_PLUGIN_API_VERSION 1

// ── Plugin Manifest ──
typedef struct {
    uint32_t api_version;
    const char* plugin_id;
    const char* version;
    uint32_t hooks;  // bitmask: 1<<on_tracked_object, 1<<on_frame, etc.
} CamPluginManifest;

// ── Event ──
typedef struct {
    double timestamp;
    uint32_t track_id;
    uint8_t class_id;
    float bbox[4];       // x, y, w, h (normalized 0-1)
    float confidence;
    float speed;         // km/h (0 if unavailable)
    uint8_t lane;
    float heading;
    const char* features; // JSON string from extractors
} CamTrackedObject;

typedef struct {
    char type[64];
    char plugin_id[64];
    uint8_t severity;
    float confidence;
    char metadata[1024];  // JSON payload
} CamEvent;

// ── Lifecycle ──
CamPluginManifest* cam_plugin_init(void);
void cam_plugin_cleanup(void);

// ── Hooks ──
uint32_t cam_plugin_on_tracked_object(CamTrackedObject* obj, CamEvent* out, uint32_t max_events);

#ifdef __cplusplus
}
#endif

#endif // CAM_PLUGIN_H
