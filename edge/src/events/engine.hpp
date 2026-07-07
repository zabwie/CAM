#pragma once
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <dlfcn.h>
#include "pipeline.hpp"
#include "cam_plugin.h"

struct PluginHandle {
    std::string id;
    std::string version;
    void* handle = nullptr;
    CamPluginManifest* manifest = nullptr;
};

class EventEngine {
public:
    EventEngine();
    ~EventEngine();

    bool load_plugin(const std::string& so_path);
    void unload_plugin(const std::string& plugin_id);
    void process_tracked_objects(const std::vector<TrackedObject>& objects);

    // Framework events (not plugin-based, built-in)
    using EventCallback = std::function<void(const std::string& type, const json& payload)>;
    void set_event_callback(EventCallback cb) { callback_ = std::move(cb); }

private:
    std::vector<std::unique_ptr<PluginHandle>> plugins_;
    EventCallback callback_;
};
