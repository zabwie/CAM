cmake_minimum_required(VERSION 3.20)
project(cam-edge VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ponytail: minimal deps for v1. OpenCV and nlohmann_json are the only hard requirements.
find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(CURL REQUIRED)

add_library(cam_lib
  src/pipeline.cpp
  src/device.cpp
  src/inference/backend.cpp
  src/vision/tracker.cpp
  src/vision/extractors.cpp
  src/events/engine.cpp
  src/events/plugins/speed.cpp
  src/events/plugins/wrong_way.cpp
  src/calibration.cpp
  src/scheduler.cpp
  src/sync.cpp
  src/storage.cpp
)

target_include_directories(cam_lib PUBLIC include src)
target_link_libraries(cam_lib PRIVATE OpenCV::opencv_core OpenCV::opencv_imgproc OpenCV::opencv_videoio nlohmann_json::nlohmann_json CURL::libcurl)

add_executable(cam-edge src/main.cpp)
target_link_libraries(cam-edge PRIVATE cam_lib)
