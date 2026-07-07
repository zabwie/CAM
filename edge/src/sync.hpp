#pragma once
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class SyncClient {
public:
    SyncClient(const std::string& server_url, const std::string& device_id);
    ~SyncClient();

    void send_event(json event);
    void send_telemetry(json telemetry);
    bool upload_clip(const std::string& clip_path);

    void start();
    void stop();

private:
    std::string server_url_;
    std::string device_id_;
    std::queue<json> event_queue_;
    std::queue<json> telemetry_queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    std::atomic<bool> running_{false};

    void worker_loop();
    bool post_json(const std::string& path, const json& body);
    bool post_binary(const std::string& path, const std::string& filepath);
};
