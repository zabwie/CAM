#pragma once
#include <string>
#include <vector>

enum class GpuVendor { CUDA, ROCm, OpenCL, None };
enum class CpuArch { X86_64, ARM, Unknown };

struct DeviceCapabilities {
    GpuVendor gpu_vendor = GpuVendor::None;
    std::string gpu_name;
    size_t gpu_memory_mb = 0;
    CpuArch cpu_arch = CpuArch::Unknown;
    size_t total_memory_mb = 0;
    float tflops = 0;
    bool has_npu = false;
};

class DeviceManager {
public:
    DeviceManager();
    const DeviceCapabilities& capabilities() const { return caps_; }
    std::string describe() const;

private:
    DeviceCapabilities caps_;
    void probe();
};
