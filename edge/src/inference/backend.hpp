#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "pipeline.hpp"

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;
    virtual bool load_model(const std::string& path) = 0;
    virtual std::vector<Detection> infer(const cv::Mat& frame) = 0;
    virtual void set_batch_size(int n) = 0;
    virtual std::string name() const = 0;
};

// ── TensorRT Backend (stub) ──
class TensorRTBackend : public InferenceBackend {
public:
    bool load_model(const std::string& path) override;
    std::vector<Detection> infer(const cv::Mat& frame) override;
    void set_batch_size(int n) override { batch_size_ = n; }
    std::string name() const override { return "TensorRT"; }

private:
    int batch_size_ = 1;
    // ponytail: TensorRT engine handle goes here
    void* engine_ = nullptr;
};

// ── ONNX Runtime Backend (stub) ──
class ONNXBackend : public InferenceBackend {
public:
    bool load_model(const std::string& path) override;
    std::vector<Detection> infer(const cv::Mat& frame) override;
    void set_batch_size(int n) override { batch_size_ = n; }
    std::string name() const override { return "ONNX Runtime"; }

private:
    int batch_size_ = 1;
    void* session_ = nullptr;
};

// ── Dummy Backend (for testing without GPU) ──
class DummyBackend : public InferenceBackend {
public:
    bool load_model(const std::string& path) override;
    std::vector<Detection> infer(const cv::Mat& frame) override;
    void set_batch_size(int n) override {}
    std::string name() const override { return "Dummy"; }
};
