cmake_minimum_required(VERSION 3.14)
project(TrajectoryPlannerInference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime路径 - 根据实际安装位置修改
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
else()
    # 尝试多个常见路径
    set(ONNXRUNTIME_ROOT_SEARCH_PATHS
        /usr/local
        /usr
        /opt/onnxruntime
        $ENV{CONDA_PREFIX}
    )

    foreach(SEARCH_PATH ${ONNXRUNTIME_ROOT_SEARCH_PATHS})
        if(EXISTS "${SEARCH_PATH}/include/onnxruntime_cxx_api.h")
            set(ONNXRUNTIME_ROOT ${SEARCH_PATH})
            break()
        endif()
    endforeach()
endif()

# 查找ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
        ${ONNXRUNTIME_ROOT}/include
        ${ONNXRUNTIME_ROOT}
        /usr/local/include
        /usr/include
        /opt/onnxruntime/include
        $ENV{CONDA_PREFIX}/include
    NO_DEFAULT_PATH
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS
        ${ONNXRUNTIME_ROOT}/lib
        ${ONNXRUNTIME_ROOT}
        /usr/local/lib
        /usr/lib
        /opt/onnxruntime/lib
        $ENV{CONDA_PREFIX}/lib
    NO_DEFAULT_PATH
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(WARNING "找不到ONNX Runtime，尝试继续构建...")
    message(WARNING "请安装ONNX Runtime或设置ONNXRUNTIME_ROOT环境变量")
    message(WARNING "pip install onnxruntime")
    message(WARNING "然后设置 export ONNXRUNTIME_ROOT=$CONDA_PREFIX")
else()
    message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
endif()

# 创建可执行文件
add_executable(inference
    src/inference.cpp
)

# 包含头文件
target_include_directories(inference PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
)

# 链接库
if(ONNXRUNTIME_LIBRARY)
    target_link_libraries(inference
        ${ONNXRUNTIME_LIBRARY}
    )
endif()

# 线程库
find_package(Threads REQUIRED)
target_link_libraries(inference Threads::Threads)

# 安装规则
install(TARGETS inference DESTINATION bin)

# 打印构建信息
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
