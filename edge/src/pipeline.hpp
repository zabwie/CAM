#pragma once
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <opencv2/core.hpp>

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t max_size) : max_size_(max_size) {}
    void push(T item) {
        std::unique_lock lock(mutex_);
        not_full_.wait(lock, [this] { return queue_.size() < max_size_ || done_; });
        if (done_) return;
        queue_.push(std::move(item));
        not_empty_.notify_one();
    }
    bool pop(T& item) {
        std::unique_lock lock(mutex_);
        not_empty_.wait(lock, [this] { return !queue_.empty() || done_; });
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return true;
    }
    void done() {
        std::lock_guard lock(mutex_);
        done_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    size_t max_size_;
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    bool done_ = false;
};

struct Frame {
    cv::Mat image;
    double timestamp;
    int64_t frame_num;
};

struct Detection {
    float x, y, w, h;
    int class_id;
    float confidence;
};

struct TrackedObject {
    int track_id;
    int class_id;
    float confidence;
    float bbox[4];
    float speed = 0;
    uint8_t lane = 0;
    float heading = 0;
    std::vector<std::tuple<float, float, double>> trajectory; // x, y, timestamp
};

using FrameQueue = BoundedQueue<Frame>;
using DetectionQueue = BoundedQueue<std::vector<Detection>>;
using TrackedObjectQueue = BoundedQueue<std::vector<TrackedObject>>;
