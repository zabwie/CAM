#include "backend.hpp"
#include <random>
#include <iostream>

// ── TensorRT stub ──
bool TensorRTBackend::load_model(const std::string& path) {
    std::cout << "[TensorRT] Loading model: " << path << std::endl;
    // ponytail: real impl deserializes TensorRT engine from .engine file
    return true;
}

std::vector<Detection> TensorRTBackend::infer(const cv::Mat& frame) {
    // ponytail: mock inference for now
    std::vector<Detection> dets;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> pos(0.1f, 0.9f);
    std::uniform_real_distribution<float> size(0.02f, 0.15f);

    for (int i = 0; i < 3; i++) {
        Detection d;
        d.x = pos(rng);
        d.y = pos(rng);
        d.w = size(rng);
        d.h = size(rng);
        d.class_id = i % 3;
        d.confidence = 0.7f + pos(rng) * 0.3f;
        dets.push_back(d);
    }
    return dets;
}

// ── ONNX stub ──
bool ONNXBackend::load_model(const std::string& path) {
    std::cout << "[ONNX] Loading model: " << path << std::endl;
    return true;
}

std::vector<Detection> ONNXBackend::infer(const cv::Mat& frame) {
    return {}; // fallback — no mock needed
}

// ── Dummy backend ──
bool DummyBackend::load_model(const std::string& path) {
    std::cout << "[Dummy] Loaded (no actual inference): " << path << std::endl;
    return true;
}

std::vector<Detection> DummyBackend::infer(const cv::Mat& frame) {
    // Return fixed detections for testing
    std::vector<Detection> dets;
    for (int i = 0; i < 2; i++) {
        Detection d;
        d.x = 0.2f + i * 0.3f;
        d.y = 0.4f;
        d.w = 0.1f;
        d.h = 0.2f;
        d.class_id = 2; // car
        d.confidence = 0.95f;
        dets.push_back(d);
    }
    return dets;
}
