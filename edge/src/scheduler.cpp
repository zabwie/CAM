#include "scheduler.hpp"
#include <iostream>
#include <algorithm>

AiScheduler::AiScheduler(int frame_budget, int device_cap_tflops)
    : frame_budget_(frame_budget), device_cap_tflops_(device_cap_tflops) {}

bool AiScheduler::load_config(const json& config) {
    if (!config.contains("allocation")) return false;

    frame_budget_ = config.value("frame_budget", 30);
    critical_slots_ = 0;
    normal_slots_ = 0;
    low_slots_ = 0;

    for (const auto& [id, alloc] : config["allocation"].items()) {
        PluginBudget budget;
        budget.plugin_id = id;
        std::string prio = alloc.value("priority", "normal");
        if (prio == "critical") { budget.priority = PluginPriority::Critical; budget.reserved_fps = alloc.value("reserved_fps", 0); }
        else if (prio == "normal") budget.priority = PluginPriority::Normal;
        else if (prio == "low") budget.priority = PluginPriority::Low;
        else budget.priority = PluginPriority::Idle;
        budget.weight = alloc.value("weight", 1.0f);
        allocations_[id] = budget;
        frame_counters_[id] = 0;

        switch (budget.priority) {
            case PluginPriority::Critical: critical_slots_ += budget.reserved_fps; break;
            case PluginPriority::Normal: normal_slots_ += static_cast<int>(budget.weight); break;
            case PluginPriority::Low: low_slots_ += 1; break;
            default: break;
        }
    }

    // Warn if over-provisioned
    int total_required = critical_slots_ + normal_slots_ + low_slots_;
    if (total_required > frame_budget_) {
        std::cout << "[scheduler] WARNING: total budget " << total_required
                  << " exceeds frame budget " << frame_budget_ << std::endl;
    }

    std::cout << "[scheduler] Loaded config: critical=" << critical_slots_
              << " normal=" << normal_slots_ << " low=" << low_slots_
              << " budget=" << frame_budget_ << " FPS" << std::endl;
    return true;
}

bool AiScheduler::should_process(const std::string& plugin_id, int frame_num) {
    auto it = allocations_.find(plugin_id);
    if (it == allocations_.end()) return true;

    const auto& budget = it->second;
    int& counter = frame_counters_[plugin_id];

    switch (budget.priority) {
        case PluginPriority::Critical: {
            // Guaranteed slot every N frames
            if (budget.reserved_fps <= 0) return true;
            int interval = frame_budget_ / budget.reserved_fps;
            return (frame_num % std::max(interval, 1)) == (counter++ % std::max(interval, 1));
        }
        case PluginPriority::Normal: {
            // Proportional share of remaining budget
            int remaining = frame_budget_ - critical_slots_;
            if (remaining <= 0) return false;
            int interval = static_cast<int>(remaining / std::max(budget.weight, 0.1f));
            return (frame_num % std::max(interval, 1)) == 0;
        }
        case PluginPriority::Low: {
            // Opportunistic
            return (frame_num % (frame_budget_ * 2)) == 0;
        }
        case PluginPriority::Idle: {
            // Only when nothing else needs it
            return (frame_num % (frame_budget_ * 10)) == 0;
        }
    }
    return true;
}

json AiScheduler::status() const {
    json j;
    j["frame_budget"] = frame_budget_;
    j["allocations"] = json::object();
    for (const auto& [id, budget] : allocations_) {
        std::string prio;
        switch (budget.priority) {
            case PluginPriority::Critical: prio = "critical"; break;
            case PluginPriority::Normal: prio = "normal"; break;
            case PluginPriority::Low: prio = "low"; break;
            case PluginPriority::Idle: prio = "idle"; break;
        }
        j["allocations"][id] = {{"priority", prio}, {"reserved_fps", budget.reserved_fps}, {"weight", budget.weight}};
    }
    return j;
}
