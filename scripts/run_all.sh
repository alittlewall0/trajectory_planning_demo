#!/bin/bash

# 轨迹规划模型完整流程脚本
# 从数据生成 -> 训练 -> ONNX导出 -> C++推理

set -e  # 遇到错误时退出

echo "=========================================="
echo "  轨迹规划模型 Demo"
echo "  完整流程"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 步骤1: 生成数据
echo -e "${YELLOW}步骤 1: 生成训练数据${NC}"
echo "----------------------------------------"
python src/dataset.py
echo -e "${GREEN}✓ 数据生成完成${NC}"
echo ""

# 步骤2: 训练模型
echo -e "${YELLOW}步骤 2: 训练模型${NC}"
echo "----------------------------------------"
python src/train.py --epochs 50 --batch_size 32 --model_type improved
echo -e "${GREEN}✓ 模型训练完成${NC}"
echo ""

# 步骤3: 导出ONNX
echo -e "${YELLOW}步骤 3: 导出ONNX模型${NC}"
echo "----------------------------------------"
python src/export_onnx.py --model_path models/best_model.pth --model_type improved
echo -e "${GREEN}✓ ONNX导出完成${NC}"
echo ""

# 步骤4: 编译C++推理代码
echo -e "${YELLOW}步骤 4: 编译C++推理代码${NC}"
echo "----------------------------------------"
if [ ! -d "build" ]; then
    mkdir build
fi
cd build
cmake ..
make -j$(nproc)
cd ..
echo -e "${GREEN}✓ 编译完成${NC}"
echo ""

# 步骤5: 运行C++推理
echo -e "${YELLOW}步骤 5: 运行C++推理${NC}"
echo "----------------------------------------"
./build/inference models/trajectory_planner.onnx
echo -e "${GREEN}✓ 推理完成${NC}"
echo ""

# 步骤6: 可视化结果
echo -e "${YELLOW}步骤 6: 可视化结果${NC}"
echo "----------------------------------------"
if [ -f "visualize_cpp_output.py" ]; then
    python visualize_cpp_output.py
    echo -e "${GREEN}✓ 可视化完成${NC}"
fi
echo ""

echo "=========================================="
echo -e "${GREEN}完整流程执行完成!${NC}"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  - 数据: data/train_data.pkl"
echo "  - 模型: models/best_model.pth"
echo "  - ONNX: models/trajectory_planner.onnx"
echo "  - 可执行文件: build/inference"
echo "  - 结果图: cpp_inference_result.png"
echo ""
