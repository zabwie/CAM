#pragma once
#include <string>
#include <sqlite3.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class LocalStorage {
public:
    explicit LocalStorage(const std::string& db_path);
    ~LocalStorage();

    bool init();
    void store_event(const json& event);
    void store_telemetry(const json& telemetry);
    std::vector<json> get_pending_events(int limit = 100);
    void mark_synced(const std::string& event_id);

    // Circular video buffer
    void write_clip(const std::string& clip_id, const uint8_t* data, size_t size);
    std::string get_clip_path(const std::string& clip_id) const;

private:
    std::string db_path_;
    sqlite3* db_ = nullptr;
    std::string data_dir_;

    bool exec(const std::string& sql);
};
