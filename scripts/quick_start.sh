#!/bin/bash

# 快速开始脚本 - 仅生成数据并使用预训练模型推理

echo "=========================================="
echo "  快速开始 - 轨迹规划Demo"
echo "=========================================="
echo ""

# 检查是否有预训练模型
if [ ! -f "models/trajectory_planner.onnx" ]; then
    echo "未找到预训练模型，需要完整训练..."
    echo "请运行: bash scripts/run_all.sh"
    exit 1
fi

echo "使用预训练模型进行推理..."
echo ""

# 检查是否已编译
if [ ! -f "build/inference" ]; then
    echo "编译C++推理代码..."
    mkdir -p build
    cd build
    cmake ..
    make -j$(nproc)
    cd ..
fi

# 运行推理
echo "运行推理..."
./build/inference models/trajectory_planner.onnx

# 可视化
if [ -f "visualize_cpp_output.py" ]; then
    echo ""
    echo "生成可视化..."
    python visualize_cpp_output.py
fi

echo ""
echo "完成!"
echo "查看结果: cpp_inference_result.png"
