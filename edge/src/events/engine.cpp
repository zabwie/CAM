#include "engine.hpp"
#include <iostream>

EventEngine::EventEngine() = default;

EventEngine::~EventEngine() {
    for (auto& p : plugins_) {
        if (p->handle) {
            if (p->manifest) {
                auto cleanup = reinterpret_cast<void(*)()>(dlsym(p->handle, "cam_plugin_cleanup"));
                if (cleanup) cleanup();
            }
            dlclose(p->handle);
        }
    }
}

bool EventEngine::load_plugin(const std::string& so_path) {
    void* handle = dlopen(so_path.c_str(), RTLD_NOW);
    if (!handle) {
        std::cerr << "[events] Failed to load plugin " << so_path << ": " << dlerror() << std::endl;
        return false;
    }

    auto init = reinterpret_cast<CamPluginManifest*(*)()>(dlsym(handle, "cam_plugin_init"));
    if (!init) {
        std::cerr << "[events] No cam_plugin_init symbol in " << so_path << std::endl;
        dlclose(handle);
        return false;
    }

    auto manifest = init();
    if (!manifest || manifest->api_version != CAM_PLUGIN_API_VERSION) {
        std::cerr << "[events] Plugin API version mismatch in " << so_path << std::endl;
        dlclose(handle);
        return false;
    }

    auto plugin = std::make_unique<PluginHandle>();
    plugin->id = manifest->plugin_id;
    plugin->version = manifest->version;
    plugin->handle = handle;
    plugin->manifest = manifest;
    plugins_.push_back(std::move(plugin));

    std::cout << "[events] Loaded plugin: " << manifest->plugin_id << " v" << manifest->version << std::endl;
    return true;
}

void EventEngine::unload_plugin(const std::string& plugin_id) {
    for (auto it = plugins_.begin(); it != plugins_.end(); ++it) {
        if ((*it)->id == plugin_id) {
            if ((*it)->handle) {
                auto cleanup = reinterpret_cast<void(*)()>(dlsym((*it)->handle, "cam_plugin_cleanup"));
                if (cleanup) cleanup();
                dlclose((*it)->handle);
            }
            plugins_.erase(it);
            std::cout << "[events] Unloaded plugin: " << plugin_id << std::endl;
            return;
        }
    }
}

void EventEngine::process_tracked_objects(const std::vector<TrackedObject>& objects) {
    for (auto& plugin : plugins_) {
        auto on_obj = reinterpret_cast<uint32_t(*)(CamTrackedObject*, CamEvent*, uint32_t)>(
            dlsym(plugin->handle, "cam_plugin_on_tracked_object"));
        if (!on_obj) continue;

        for (const auto& obj : objects) {
            CamTrackedObject cam_obj;
            cam_obj.timestamp = obj.trajectory.empty() ? 0 : std::get<2>(obj.trajectory.back());
            cam_obj.track_id = obj.track_id;
            cam_obj.class_id = static_cast<uint8_t>(obj.class_id);
            cam_obj.bbox[0] = obj.bbox[0]; cam_obj.bbox[1] = obj.bbox[1];
            cam_obj.bbox[2] = obj.bbox[2]; cam_obj.bbox[3] = obj.bbox[3];
            cam_obj.confidence = obj.confidence;
            cam_obj.speed = obj.speed;
            cam_obj.lane = obj.lane;
            cam_obj.heading = obj.heading;
            cam_obj.features = "";

            CamEvent events[8];
            uint32_t count = on_obj(&cam_obj, events, 8);
            for (uint32_t i = 0; i < count; i++) {
                if (callback_) {
                    json payload = json::parse(events[i].metadata, nullptr, false);
                    if (payload.is_discarded()) payload = json::object();
                    callback_(events[i].type, payload);
                }
            }
        }
    }
}
