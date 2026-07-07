#pragma once
#include <string>
#include <map>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

enum class PluginPriority { Critical, Normal, Low, Idle };

struct PluginBudget {
    std::string plugin_id;
    PluginPriority priority = PluginPriority::Normal;
    int reserved_fps = 0;
    float weight = 1.0f;
};

class AiScheduler {
public:
    AiScheduler(int frame_budget = 30, int device_cap_tflops = 0);

    bool load_config(const json& config);
    bool should_process(const std::string& plugin_id, int frame_num);
    void set_frame_budget(int fps) { frame_budget_ = fps; }
    json status() const;

private:
    int frame_budget_;
    int device_cap_tflops_;
    std::map<std::string, PluginBudget> allocations_;
    std::map<std::string, int> frame_counters_;

    int critical_slots_ = 0;
    int normal_slots_ = 0;
    int low_slots_ = 0;
};
