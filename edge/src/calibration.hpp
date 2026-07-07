#pragma once
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct Calibration {
    std::string id;
    std::string device_id;
    int version = 1;
    std::vector<std::vector<double>> homography; // 3x3 perspective transform
    json lanes;
    json zones;
    double confidence = 0.0;
    std::string created_by = "auto";
    std::string created_at;
    bool active = true;
};

class CalibrationService {
public:
    CalibrationService(const std::string& db_path);
    bool load_active();
    const Calibration& current() const { return cal_; }
    bool apply_calibration(const json& doc);
    double pixels_per_meter() const;

private:
    std::string db_path_;
    Calibration cal_;
    bool loaded_ = false;
};
