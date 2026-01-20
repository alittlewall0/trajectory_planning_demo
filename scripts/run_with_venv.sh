#!/bin/bash

# 激活虚拟环境并运行完整流程

set -e

echo "=========================================="
echo "  轨迹规划Demo - 虚拟环境版"
echo "=========================================="
echo ""

# 获取项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "错误: 虚拟环境不存在"
    echo "请先运行: bash scripts/setup_ubuntu.sh"
    exit 1
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate
echo "✓ 虚拟环境已激活: $(which python)"
echo ""

# 检查ONNX Runtime
if [ -z "$ONNXRUNTIME_ROOT" ]; then
    if [ -d "onnxruntime-linux-x64-1.16.0" ]; then
        export ONNXRUNTIME_ROOT="$PROJECT_ROOT/onnxruntime-linux-x64-1.16.0"
        export LD_LIBRARY_PATH="$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH"
        echo "✓ 设置ONNX Runtime路径"
    fi
fi

# 执行完整流程
echo ""
echo "=========================================="
echo "  执行完整流程"
echo "=========================================="
echo ""

# 步骤1: 生成数据
echo -e "\033[1;33m步骤 1: 生成训练数据\033[0m"
echo "----------------------------------------"
python src/dataset.py
echo -e "\033[0;32m✓ 数据生成完成\033[0m"
echo ""

# 步骤2: 训练模型
echo -e "\033[1;33m步骤 2: 训练模型\033[0m"
echo "----------------------------------------"
python src/train.py --epochs 50 --batch_size 32 --model_type improved
echo -e "\033[0;32m✓ 模型训练完成\033[0m"
echo ""

# 步骤3: 导出ONNX
echo -e "\033[1;33m步骤 3: 导出ONNX模型\033[0m"
echo "----------------------------------------"
python src/export_onnx.py --model_path models/best_model.pth --model_type improved
echo -e "\033[0;32m✓ ONNX导出完成\033[0m"
echo ""

# 步骤4: 编译C++推理代码
echo -e "\033[1;33m步骤 4: 编译C++推理代码\033[0m"
echo "----------------------------------------"
if [ ! -d "build" ]; then
    mkdir build
fi
cd build
cmake ..
make -j$(nproc)
cd ..
echo -e "\033[0;32m✓ 编译完成\033[0m"
echo ""

# 步骤5: 运行C++推理
echo -e "\033[1;33m步骤 5: 运行C++推理\033[0m"
echo "----------------------------------------"
./build/inference models/trajectory_planner.onnx
echo -e "\033[0;32m✓ 推理完成\033[0m"
echo ""

# 步骤6: 可视化结果
echo -e "\033[1;33m步骤 6: 可视化结果\033[0m"
echo "----------------------------------------"
if [ -f "visualize_cpp_output.py" ]; then
    python visualize_cpp_output.py
    echo -e "\033[0;32m✓ 可视化完成\033[0m"
fi
echo ""

echo "=========================================="
echo -e "\033[0;32m完整流程执行完成!\033[0m"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  - 数据: data/train_data.pkl"
echo "  - 模型: models/best_model.pth"
echo "  - ONNX: models/trajectory_planner.onnx"
echo "  - 可执行文件: build/inference"
echo "  - 结果图: cpp_inference_result.png"
echo ""
echo "提示: 以后只需运行 'bash scripts/run_with_venv.sh'"
echo ""
