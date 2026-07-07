#include "cam_plugin.h"
#include <cstring>
#include <cstdio>

static CamPluginManifest manifest = {
    CAM_PLUGIN_API_VERSION,
    "speed",
    "0.1.0",
    1 // on_tracked_object
};

extern "C" CamPluginManifest* cam_plugin_init() {
    return &manifest;
}

extern "C" void cam_plugin_cleanup() {}

extern "C" uint32_t cam_plugin_on_tracked_object(CamTrackedObject* obj, CamEvent* out, uint32_t max_events) {
    if (max_events < 1) return 0;

    // Speed threshold: 45 km/h default
    if (obj->speed < 1.0f) return 0; // not moving or no speed data

    snprintf(out->type, sizeof(out->type), "traffic.speed");
    snprintf(out->plugin_id, sizeof(out->plugin_id), "speed");
    out->severity = (obj->speed > 80.0f) ? 2 : (obj->speed > 45.0f) ? 1 : 0;
    out->confidence = obj->confidence;

    snprintf(out->metadata, sizeof(out->metadata),
        R"({"speed_kmh":%.1f,"vehicle_class":"car","lane":%d,"direction":"N"})",
        obj->speed, obj->lane);

    return 1;
}
