#include "sync.hpp"
#include <iostream>
#include <curl/curl.h>
#include <fstream>
#include <sstream>

static size_t write_callback(char* data, size_t size, size_t nmemb, void*) {
    return size * nmemb;
}

SyncClient::SyncClient(const std::string& server_url, const std::string& device_id)
    : server_url_(server_url), device_id_(device_id) {
    curl_global_init(CURL_GLOBAL_ALL);
}

SyncClient::~SyncClient() {
    stop();
    curl_global_cleanup();
}

void SyncClient::start() {
    running_ = true;
    worker_ = std::thread(&SyncClient::worker_loop, this);
    std::cout << "[sync] Started, target: " << server_url_ << std::endl;
}

void SyncClient::stop() {
    running_ = false;
    cv_.notify_one();
    if (worker_.joinable()) worker_.join();
}

void SyncClient::send_event(json event) {
    event["device_id"] = device_id_;
    std::lock_guard lock(mutex_);
    event_queue_.push(std::move(event));
    cv_.notify_one();
}

void SyncClient::send_telemetry(json telemetry) {
    telemetry["device_id"] = device_id_;
    std::lock_guard lock(mutex_);
    telemetry_queue_.push(std::move(telemetry));
}

bool SyncClient::upload_clip(const std::string& clip_path) {
    return post_binary("/api/v1/media/upload", clip_path);
}

bool SyncClient::post_json(const std::string& path, const json& body) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;

    std::string url = server_url_ + path;
    std::string data = body.dump();

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, data.size());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::cerr << "[sync] POST failed: " << curl_easy_strerror(res) << std::endl;
        return false;
    }
    return true;
}

bool SyncClient::post_binary(const std::string& path, const std::string& filepath) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;

    std::string url = server_url_ + path;
    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) { std::cerr << "[sync] Cannot open " << filepath << std::endl; return false; }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_READDATA, f);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, size);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/octet-stream");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    fclose(f);
    curl_easy_cleanup(curl);

    return res == CURLE_OK;
}

void SyncClient::worker_loop() {
    while (running_) {
        std::unique_lock lock(mutex_);
        cv_.wait_for(lock, std::chrono::seconds(1), [this] { return !event_queue_.empty() || !running_; });

        if (!running_) break;

        // Upload events in batches
        std::vector<json> batch;
        while (!event_queue_.empty() && batch.size() < 50) {
            batch.push_back(std::move(event_queue_.front()));
            event_queue_.pop();
        }
        lock.unlock();

        if (!batch.empty()) {
            json body = {{"events", batch}};
            if (!post_json("/api/v1/ingest", body)) {
                // Re-queue on failure
                std::lock_guard relock(mutex_);
                for (auto& ev : batch) event_queue_.push(std::move(ev));
            }
        }
    }
}
