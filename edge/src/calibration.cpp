#include "calibration.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

CalibrationService::CalibrationService(const std::string& db_path) : db_path_(db_path) {}

bool CalibrationService::load_active() {
    // ponytail: loads calibration from JSON file. Real impl uses SQLite.
    std::string cal_path = db_path_ + "/calibration.json";
    std::ifstream f(cal_path);
    if (!f.is_open()) {
        std::cout << "[calibration] No calibration file found, using defaults" << std::endl;
        cal_.confidence = 0.5;
        cal_.homography = {{1,0,0}, {0,1,0}, {0,0,1}};
        return false;
    }

    try {
        json doc = json::parse(f);
        cal_.id = doc.value("id", "local");
        cal_.version = doc.value("version", 1);
        cal_.confidence = doc.value("confidence", 0.0);
        cal_.lanes = doc.value("lanes", json::object());
        cal_.zones = doc.value("zones", json::object());
        if (doc.contains("homography")) {
            cal_.homography = doc["homography"].get<std::vector<std::vector<double>>>();
        }
        loaded_ = true;
        std::cout << "[calibration] Loaded v" << cal_.version << " (confidence: " << cal_.confidence << ")" << std::endl;
    } catch (const json::parse_error& e) {
        std::cerr << "[calibration] Parse error: " << e.what() << std::endl;
        return false;
    }
    return true;
}

bool CalibrationService::apply_calibration(const json& doc) {
    cal_ = Calibration{};
    cal_.id = doc.value("id", "local");
    cal_.version = doc.value("version", 1);
    cal_.confidence = doc.value("confidence", 0.0);
    cal_.lanes = doc.value("lanes", json::object());
    cal_.zones = doc.value("zones", json::object());
    if (doc.contains("homography")) {
        cal_.homography = doc["homography"].get<std::vector<std::vector<double>>>();
    }
    loaded_ = true;

    std::string cal_path = db_path_ + "/calibration.json";
    std::ofstream f(cal_path);
    if (f.is_open()) {
        f << doc.dump(2);
        std::cout << "[calibration] Saved v" << cal_.version << std::endl;
    }
    return true;
}

double CalibrationService::pixels_per_meter() const {
    // ponytail: extract scale from homography. Default 50 px/m = ~720p camera at 15m height.
    if (cal_.homography.size() < 3) return 50.0;
    return std::abs(cal_.homography[0][0]) * 100.0;
}
