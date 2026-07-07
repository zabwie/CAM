#include "device.hpp"
#include <fstream>
#include <sstream>

DeviceManager::DeviceManager() { probe(); }

void DeviceManager::probe() {
    // ponytail: basic probing via /proc and env. Real impl uses CUDA/ROCm APIs.
    caps_.cpu_arch = CpuArch::X86_64;

    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.starts_with("MemTotal:")) {
            std::istringstream(line) >> line >> caps_.total_memory_mb;
            caps_.total_memory_mb /= 1024; // kB → MB
        }
    }

    // Check for NVIDIA GPU
    if (system("which nvidia-smi >/dev/null 2>&1") == 0) {
        caps_.gpu_vendor = GpuVendor::CUDA;
        caps_.gpu_name = "NVIDIA GPU";
    }

    caps_.has_npu = false; // NPU detection requires vendor SDKs
}

std::string DeviceManager::describe() const {
    std::string desc;
    if (caps_.gpu_vendor != GpuVendor::None)
        desc += "GPU: " + caps_.gpu_name + " (" + std::to_string(caps_.gpu_memory_mb) + " MB), ";
    desc += "RAM: " + std::to_string(caps_.total_memory_mb) + " MB";
    return desc;
}
