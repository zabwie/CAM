#include "extractors.hpp"
#include <cmath>
#include <algorithm>

float extract_speed(const TrackedObject& obj, const CalibrationData& cal) {
    if (obj.trajectory.size() < 2) return 0;

    // Use last two trajectory points
    const auto& [x1, y1, ts1] = obj.trajectory[obj.trajectory.size() - 2];
    const auto& [x2, y2, ts2] = obj.trajectory.back();

    float dt = static_cast<float>(ts2 - ts1);
    if (dt <= 0) return 0;

    // Pixel displacement
    float dx = x2 - x1;
    float dy = y2 - y1;
    float pixels_per_sec = std::sqrt(dx * dx + dy * dy) / dt;

    // Convert: pixels/sec → meters/sec → km/h
    float meters_per_sec = pixels_per_sec / cal.pixels_per_meter;
    return meters_per_sec * 3.6f;
}

uint8_t extract_lane(const TrackedObject& obj, int lane_count) {
    // ponytail: simple lane assignment by horizontal position
    float cx = obj.bbox[0] + obj.bbox[2] / 2;
    int lane = static_cast<int>(cx * lane_count);
    return static_cast<uint8_t>(std::clamp(lane, 0, lane_count - 1)) + 1;
}

float extract_heading(const TrackedObject& obj) {
    if (obj.trajectory.size() < 2) return 0;

    const auto& [x1, y1, ts1] = obj.trajectory[obj.trajectory.size() - 2];
    const auto& [x2, y2, ts2] = obj.trajectory.back();

    float dx = x2 - x1;
    float dy = y2 - y1;
    float angle = std::atan2(dy, dx) * 180.0f / static_cast<float>(M_PI);
    return std::fmod(angle + 360.0f, 360.0f);
}
