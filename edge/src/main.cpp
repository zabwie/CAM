#include "pipeline.hpp"
#include "device.hpp"
#include "calibration.hpp"
#include "scheduler.hpp"
#include "events/engine.hpp"
#include "sync.hpp"
#include "storage.hpp"
#include "inference/backend.hpp"
#include <iostream>
#include <csignal>
#include <atomic>

static std::atomic<bool> g_running{true};

void signal_handler(int) {
    g_running = false;
}

int main(int argc, char* argv[]) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    std::cout << "╔══════════════════════════════════╗" << std::endl;
    std::cout << "║        Cam Edge v0.1.0           ║" << std::endl;
    std::cout << "╚══════════════════════════════════╝" << std::endl;

    // ── Device Detection ──
    DeviceManager device;
    std::cout << "[system] " << device.describe() << std::endl;

    // ── Config from args / env ──
    std::string server_url = "http://localhost:3000";
    std::string device_id = "dev_" + std::to_string(getpid());
    std::string data_dir = "/var/cam";
    int frame_budget = 30;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--server" && i + 1 < argc) server_url = argv[++i];
        else if (arg == "--device-id" && i + 1 < argc) device_id = argv[++i];
        else if (arg == "--data-dir" && i + 1 < argc) data_dir = argv[++i];
        else if (arg == "--fps" && i + 1 < argc) frame_budget = std::stoi(argv[++i]);
    }

    // ── Storage ──
    LocalStorage storage(data_dir);
    if (!storage.init()) {
        std::cerr << "[system] Failed to initialize storage" << std::endl;
        return 1;
    }
    // ponytail: store config in SQLite when settings API is available

    // ── Calibration ──
    CalibrationService calibration(data_dir);
    calibration.load_active();
    std::cout << "[system] Calibration confidence: " << (calibration.current().confidence * 100) << "%" << std::endl;

    // ── AI Scheduler ──
    AiScheduler scheduler(frame_budget);
    json scheduler_config = {
        {"frame_budget", frame_budget},
        {"allocation", {
            {"speed", {{"priority", "critical"}, {"reserved_fps", 15}}},
            {"wrong_way", {{"priority", "critical"}, {"reserved_fps", 10}}},
            {"congestion", {{"priority", "normal"}, {"weight", 1.0}}},
            {"stopped_vehicle", {{"priority", "low"}}}
        }}
    };
    scheduler.load_config(scheduler_config);

    // ── Event Engine ──
    EventEngine events;
    events.set_event_callback([&](const std::string& type, const json& payload) {
        json event = {
            {"id", ""}, // server assigns UUID
            {"ts", std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()},
            {"device_id", device_id},
            {"type", type},
            {"plugin_id", type.substr(0, type.find('.'))},
            {"severity", payload.value("severity", 0)},
            {"confidence", payload.value("confidence", 1.0)},
            {"source", 0},
            {"metadata", payload}
        };
        storage.store_event(event);
    });

    // Try loading plugins
    events.load_plugin("plugins/speed.so");
    events.load_plugin("plugins/wrong_way.so");

    // ── Sync ──
    SyncClient sync(server_url, device_id);
    sync.start();

    // ── Pipeline ──
    Pipeline pipeline(frame_budget);
    if (argc > 1 && argv[1][0] != '-') {
        pipeline.set_input(argv[1]); // video file or RTSP URL
    }

    std::cout << "[system] Starting pipeline (device: " << device_id << ")" << std::endl;
    pipeline.start();

    // ── Main loop ──
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cout << "[system] Shutting down..." << std::endl;
    pipeline.stop();
    pipeline.join();
    sync.stop();

    std::cout << "[system] Goodbye." << std::endl;
    return 0;
}
