cmake_minimum_required(VERSION 3.14)
project(TrajectoryPlannerInference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime路径 - 根据实际安装位置修改
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
else()
    set(ONNXRUNTIME_ROOT /usr/local)
endif()

# 查找ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
        ${ONNXRUNTIME_ROOT}/include
        /usr/local/include
        /opt/onnxruntime/include
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS
        ${ONNXRUNTIME_ROOT}/lib
        /usr/local/lib
        /opt/onnxruntime/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "找不到ONNX Runtime。请设置ONNXRUNTIME_ROOT环境变量或安装ONNX Runtime。")
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

# 创建可执行文件
add_executable(inference
    src/inference.cpp
)

# 包含头文件
target_include_directories(inference PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 链接库
target_link_libraries(inference
    ${ONNXRUNTIME_LIBRARY}
)

# 线程库
find_package(Threads REQUIRED)
target_link_libraries(inference Threads::Threads)

# 安装规则
install(TARGETS inference DESTINATION bin)

# 打印构建信息
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
