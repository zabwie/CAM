#pragma once
#include "pipeline.hpp"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ── Speed Extractor ──
// Converts pixel displacement to km/h using calibration data.
struct CalibrationData {
    double pixels_per_meter = 50.0; // default — overridden by calibration doc
    double fps = 30.0;
};

float extract_speed(const TrackedObject& obj, const CalibrationData& cal);

// ── Lane Extractor ──
// Determines lane from horizontal position and lane geometry.
uint8_t extract_lane(const TrackedObject& obj, int lane_count = 2);

// ── Direction Extractor ──
// Computes heading from trajectory.
float extract_heading(const TrackedObject& obj);
