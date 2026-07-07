#include "cam_plugin.h"
#include <cstring>

static CamPluginManifest manifest = {
    CAM_PLUGIN_API_VERSION,
    "wrong_way",
    "0.1.0",
    1
};

extern "C" CamPluginManifest* cam_plugin_init() {
    return &manifest;
}

extern "C" void cam_plugin_cleanup() {}

extern "C" uint32_t cam_plugin_on_tracked_object(CamTrackedObject* obj, CamEvent* out, uint32_t max_events) {
    if (max_events < 1) return 0;

    // Detect wrong-way driving: heading opposite to traffic flow (N = 0°, S = 180°)
    // ponytail: assumes northbound flow, heading 135-225° is wrong way
    if (obj->heading < 135.0f || obj->heading > 225.0f) return 0;

    snprintf(out->type, sizeof(out->type), "safety.wrong_way");
    snprintf(out->plugin_id, sizeof(out->plugin_id), "wrong_way");
    out->severity = 2;
    out->confidence = obj->confidence;

    snprintf(out->metadata, sizeof(out->metadata),
        R"({"heading":%.1f,"lane":%d,"speed_kmh":%.1f})",
        obj->heading, obj->lane, obj->speed);

    return 1;
}
