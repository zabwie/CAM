#include "pipeline.hpp"
#include <iostream>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

class Pipeline {
public:
    Pipeline(int frame_budget = 30);
    ~Pipeline();

    void set_input(const std::string& source);
    void start();
    void stop();
    void join();

private:
    FrameQueue frame_queue_{30};
    DetectionQueue det_queue_{10};
    TrackedObjectQueue obj_queue_{10};

    std::thread capture_thread_;
    std::thread infer_thread_;
    std::thread track_thread_;

    std::atomic<bool> running_{false};
    std::string source_;
    int frame_budget_;

    void capture_loop();
    void inference_loop();
    void tracking_loop();
};

Pipeline::Pipeline(int frame_budget) : frame_budget_(frame_budget) {
    std::cout << "[pipeline] Created with frame budget: " << frame_budget_ << " FPS" << std::endl;
}

Pipeline::~Pipeline() { stop(); }

void Pipeline::set_input(const std::string& source) { source_ = source; }

void Pipeline::start() {
    running_ = true;
    capture_thread_ = std::thread(&Pipeline::capture_loop, this);
    infer_thread_ = std::thread(&Pipeline::inference_loop, this);
    track_thread_ = std::thread(&Pipeline::tracking_loop, this);
    std::cout << "[pipeline] Started all threads" << std::endl;
}

void Pipeline::stop() {
    running_ = false;
    frame_queue_.done();
    det_queue_.done();
    obj_queue_.done();
}

void Pipeline::join() {
    if (capture_thread_.joinable()) capture_thread_.join();
    if (infer_thread_.joinable()) infer_thread_.join();
    if (track_thread_.joinable()) track_thread_.join();
}

void Pipeline::capture_loop() {
    cv::VideoCapture cap;
    if (!source_.empty()) {
        cap.open(source_);
    } else {
        cap.open(0); // default camera
    }

    if (!cap.isOpened()) {
        std::cerr << "[capture] Failed to open source: " << source_ << std::endl;
        return;
    }

    // ponytail: hardware decode not configured. Add NVENC/VAAPI via GStreamer pipeline string.
    int64_t frame_num = 0;
    Frame frame;
    while (running_) {
        cap >> frame.image;
        if (frame.image.empty()) {
            std::cout << "[capture] End of stream" << std::endl;
            break;
        }
        frame.timestamp = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
        frame.frame_num = frame_num++;
        frame_queue_.push(std::move(frame));
    }
    cap.release();
    frame_queue_.done();
}

void Pipeline::inference_loop() {
    // ponytail: uses DummyBackend for now. Swap for TensorRT/ONNX in production.
    DummyBackend backend;
    if (!backend.load_model("yolov8n.engine")) {
        std::cerr << "[inference] Failed to load model" << std::endl;
        return;
    }

    Frame frame;
    while (frame_queue_.pop(frame)) {
        auto dets = backend.infer(frame.image);
        det_queue_.push(std::move(dets));
    }
    det_queue_.done();
}

void Pipeline::tracking_loop() {
    // ponytail: centroid tracker. Swap for SORT/BoT-SORT when accuracy matters.
    // Feature extractors and event engine run inline with tracking for simplicity.
    CalibrationData cal;
    CentroidTracker tracker;

    std::vector<Detection> dets;
    while (det_queue_.pop(dets)) {
        double ts = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
        auto tracked = tracker.update(dets, ts);

        // Extract features
        for (auto& obj : tracked) {
            obj.speed = extract_speed(obj, cal);
            obj.lane = extract_lane(obj);
            obj.heading = extract_heading(obj);
        }

        obj_queue_.push(std::move(tracked));
    }
    obj_queue_.done();
}
