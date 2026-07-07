#pragma once
#include <vector>
#include <map>
#include "pipeline.hpp"

// ponytail: simple centroid tracker. Swap for SORT/BoT-SORT when accuracy requirements grow.
class CentroidTracker {
public:
    CentroidTracker(float iou_threshold = 0.3f, int max_lost = 10);

    std::vector<TrackedObject> update(const std::vector<Detection>& detections, double timestamp);

private:
    int next_id_ = 0;
    float iou_threshold_;
    int max_lost_;
    std::map<int, TrackedObject> tracks_; // track_id → tracked object
    std::map<int, int> lost_count_;       // track_id → frames since last seen

    float iou(const Detection& a, const TrackedObject& b) const;
    int match(const std::vector<Detection>& detections);
};
