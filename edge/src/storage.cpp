#include "storage.hpp"
#include <iostream>
#include <fstream>
#include <sys/stat.h>

LocalStorage::LocalStorage(const std::string& db_path) : db_path_(db_path) {
    data_dir_ = db_path_ + "/data";
    mkdir(data_dir_.c_str(), 0755);
}

LocalStorage::~LocalStorage() {
    if (db_) sqlite3_close(db_);
}

bool LocalStorage::exec(const std::string& sql) {
    char* err = nullptr;
    if (sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK) {
        std::cerr << "[storage] SQL error: " << (err ? err : "unknown") << std::endl;
        sqlite3_free(err);
        return false;
    }
    return true;
}

bool LocalStorage::init() {
    std::string full_path = db_path_ + "/cam_edge.db";
    if (sqlite3_open(full_path.c_str(), &db_) != SQLITE_OK) {
        std::cerr << "[storage] Failed to open DB: " << sqlite3_errmsg(db_) << std::endl;
        return false;
    }

    exec(R"(
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            device_id TEXT NOT NULL,
            type TEXT NOT NULL,
            payload TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            synced INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    )");

    std::cout << "[storage] Initialized at " << full_path << std::endl;
    return true;
}

void LocalStorage::store_event(const json& event) {
    std::string id = event.value("id", "unknown");
    std::string type = event.value("type", "unknown");
    std::string device_id = event.value("device_id", "unknown");
    int64_t ts = event.value("ts", 0LL);

    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(db_,
        "INSERT OR REPLACE INTO events (id, device_id, type, payload, timestamp) VALUES (?, ?, ?, ?, ?)",
        -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, device_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, type.c_str(), -1, SQLITE_TRANSIENT);
    std::string payload = event.dump();
    sqlite3_bind_text(stmt, 4, payload.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 5, ts);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

std::vector<json> LocalStorage::get_pending_events(int limit) {
    std::vector<json> results;
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(db_,
        "SELECT payload FROM events WHERE synced = 0 ORDER BY timestamp ASC LIMIT ?",
        -1, &stmt, nullptr);
    sqlite3_bind_int(stmt, 1, limit);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* payload = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        if (payload) results.push_back(json::parse(payload));
    }
    sqlite3_finalize(stmt);
    return results;
}

void LocalStorage::mark_synced(const std::string& event_id) {
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(db_, "UPDATE events SET synced = 1 WHERE id = ?", -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, event_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

void LocalStorage::write_clip(const std::string& clip_id, const uint8_t* data, size_t size) {
    std::string path = data_dir_ + "/" + clip_id + ".h264";
    std::ofstream f(path, std::ios::binary);
    if (f.is_open()) {
        f.write(reinterpret_cast<const char*>(data), size);
    }
}

std::string LocalStorage::get_clip_path(const std::string& clip_id) const {
    return data_dir_ + "/" + clip_id + ".h264";
}
