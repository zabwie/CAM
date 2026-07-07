#include "tracker.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

CentroidTracker::CentroidTracker(float iou_threshold, int max_lost)
    : iou_threshold_(iou_threshold), max_lost_(max_lost) {}

float CentroidTracker::iou(const Detection& a, const TrackedObject& b) const {
    float ax1 = a.x, ay1 = a.y, ax2 = a.x + a.w, ay2 = a.y + a.h;
    float bx1 = b.bbox[0], by1 = b.bbox[1], bx2 = b.bbox[0] + b.bbox[2], by2 = b.bbox[1] + b.bbox[3];

    float xi = std::max(ax1, bx1), yi = std::max(ay1, by1);
    float xi2 = std::min(ax2, bx2), yi2 = std::min(ay2, by2);
    float inter = std::max(0.0f, xi2 - xi) * std::max(0.0f, yi2 - yi);
    float area_a = (ax2 - ax1) * (ay2 - ay1);
    float area_b = (bx2 - bx1) * (by2 - by1);
    float union_area = area_a + area_b - inter;
    return union_area > 0 ? inter / union_area : 0;
}

int CentroidTracker::match(const std::vector<Detection>& detections) {
    int matched = 0;
    for (auto& [tid, track] : tracks_) {
        float best_iou = 0;
        int best_det = -1;
        const auto& last_bbox = track.bbox;

        for (size_t j = 0; j < detections.size(); j++) {
            Detection d;
            d.x = last_bbox[0]; d.y = last_bbox[1];
            d.w = last_bbox[2]; d.h = last_bbox[3];

            // simplifed: match by position proximity
            float dx = detections[j].x - last_bbox[0];
            float dy = detections[j].y - last_bbox[1];
            float dist = std::sqrt(dx * dx + dy * dy);
            float iou_val = 1.0f - std::min(dist * 2.0f, 1.0f);

            if (iou_val > best_iou) {
                best_iou = iou_val;
                best_det = static_cast<int>(j);
            }
        }

        if (best_iou >= iou_threshold_ && best_det >= 0) {
            const auto& d = detections[best_det];
            track.bbox[0] = d.x; track.bbox[1] = d.y;
            track.bbox[2] = d.w; track.bbox[3] = d.h;
            track.class_id = d.class_id;
            track.confidence = d.confidence;
            lost_count_[tid] = 0;
            matched++;
        } else {
            lost_count_[tid]++;
        }
    }
    return matched;
}

std::vector<TrackedObject> CentroidTracker::update(const std::vector<Detection>& detections, double timestamp) {
    // Match existing tracks
    int matched = match(detections);

    // Create new tracks for unmatched detections
    if (matched < static_cast<int>(detections.size())) {
        for (const auto& d : detections) {
            bool matched_det = false;
            for (const auto& [tid, track] : tracks_) {
                float dx = d.x - track.bbox[0];
                float dy = d.y - track.bbox[1];
                if (std::sqrt(dx * dx + dy * dy) < iou_threshold_) {
                    matched_det = true;
                    break;
                }
            }
            if (!matched_det) {
                TrackedObject obj;
                obj.track_id = next_id_++;
                obj.class_id = d.class_id;
                obj.confidence = d.confidence;
                obj.bbox[0] = d.x; obj.bbox[1] = d.y;
                obj.bbox[2] = d.w; obj.bbox[3] = d.h;
                obj.trajectory.emplace_back(d.x + d.w / 2, d.y + d.h / 2, timestamp);
                tracks_[obj.track_id] = obj;
                lost_count_[obj.track_id] = 0;
            }
        }
    }

    // Remove lost tracks
    std::vector<int> to_remove;
    for (const auto& [tid, count] : lost_count_) {
        if (count > max_lost_) to_remove.push_back(tid);
    }
    for (int tid : to_remove) {
        tracks_.erase(tid);
        lost_count_.erase(tid);
    }

    // Update trajectories
    for (auto& [tid, track] : tracks_) {
        track.trajectory.emplace_back(
            track.bbox[0] + track.bbox[2] / 2,
            track.bbox[1] + track.bbox[3] / 2,
            timestamp
        );
    }

    std::vector<TrackedObject> result;
    for (const auto& [tid, track] : tracks_) {
        result.push_back(track);
    }
    return result;
}
