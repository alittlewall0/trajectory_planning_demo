#!/bin/bash

# Ubuntu 24.04 环境配置脚本
# 配置Python虚拟环境和安装所有依赖

set -e

echo "=========================================="
echo "  Ubuntu 24.04 环境配置"
echo "  轨迹规划Demo项目"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${YELLOW}步骤 1: 安装系统依赖${NC}"
echo "----------------------------------------"

# 更新包列表
sudo apt update

# 安装Python虚拟环境和构建工具
sudo apt install -y python3-venv python3-dev build-essential cmake git

echo -e "${GREEN}✓ 系统依赖安装完成${NC}"
echo ""

# 步骤2: 创建虚拟环境
echo -e "${YELLOW}步骤 2: 创建Python虚拟环境${NC}"
echo "----------------------------------------"

if [ -d "venv" ]; then
    echo "虚拟环境已存在，跳过创建"
else
    python3 -m venv venv
    echo -e "${GREEN}✓ 虚拟环境创建完成${NC}"
fi
echo ""

# 步骤3: 激活虚拟环境并安装Python依赖
echo -e "${YELLOW}步骤 3: 安装Python依赖${NC}"
echo "----------------------------------------"

source venv/bin/activate

# 升级pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt

echo -e "${GREEN}✓ Python依赖安装完成${NC}"
echo ""

# 步骤4: 安装ONNX Runtime
echo -e "${YELLOW}步骤 4: 安装ONNX Runtime${NC}"
echo "----------------------------------------"

echo "选择ONNX Runtime安装方式:"
echo "1. Conda (需要安装conda)"
echo "2. Pip (仅Python库，C++库需额外下载)"
echo "3. 下载预编译版本 (推荐)"
echo "4. 跳过"
read -p "请选择 [1-4]: " onnx_choice

case $onnx_choice in
    1)
        if command -v conda &> /dev/null; then
            conda install -c conda-forge onnxruntime -y
            export ONNXRUNTIME_ROOT=$CONDA_PREFIX
            echo -e "${GREEN}✓ ONNX Runtime (Conda) 安装完成${NC}"
        else
            echo -e "${RED}✗ 未找到Conda，请先安装Miniconda或Anaconda${NC}"
        fi
        ;;

    2)
        pip install onnxruntime
        echo -e "${YELLOW}⚠ 已安装Python版本，需要额外下载C++库${NC}"
        echo "请运行: bash scripts/install_onnxruntime.sh"
        ;;

    3)
        bash scripts/install_onnxruntime.sh
        ;;

    4)
        echo -e "${YELLOW}跳过ONNX Runtime安装${NC}"
        ;;

    *)
        echo -e "${RED}无效选择${NC}"
        ;;
esac
echo ""

# 步骤5: 创建必要的目录
echo -e "${YELLOW}步骤 5: 创建项目目录${NC}"
echo "----------------------------------------"

mkdir -p data models build include

echo -e "${GREEN}✓ 目录创建完成${NC}"
echo ""

# 步骤6: 保存环境变量配置
echo -e "${YELLOW}步骤 6: 配置环境变量${NC}"
echo "----------------------------------------"

# 创建环境变量文件
cat > venv/bin/project_setup.sh << EOF
#!/bin/bash
# 项目环境变量配置

# 虚拟环境激活
export VIRTUAL_ENV="$PROJECT_ROOT/venv"
PATH="\$VIRTUAL_ENV/bin:\$PATH"

# ONNX Runtime路径 (根据实际安装位置修改)
if [ -f "onnxruntime-linux-x64-1.16.0/lib/libonnxruntime.so" ]; then
    export ONNXRUNTIME_ROOT="$PROJECT_ROOT/onnxruntime-linux-x64-1.16.0"
    export LD_LIBRARY_PATH="\$ONNXRUNTIME_ROOT/lib:\$LD_LIBRARY_PATH"
    echo "✓ ONNX Runtime: \$ONNXRUNTIME_ROOT"
fi

# 项目根目录
export PROJECT_ROOT="$PROJECT_ROOT"

echo "项目环境已激活"
echo "项目路径: \$PROJECT_ROOT"
echo "虚拟环境: \$VIRTUAL_ENV"
EOF

chmod +x venv/bin/project_setup.sh

echo -e "${GREEN}✓ 环境配置文件创建完成${NC}"
echo ""

# 完成提示
echo "=========================================="
echo -e "${GREEN}环境配置完成!${NC}"
echo "=========================================="
echo ""
echo "使用方法:"
echo ""
echo "1. 激活虚拟环境:"
echo "   source venv/bin/activate"
echo ""
echo "2. 运行完整流程:"
echo "   bash scripts/run_all.sh"
echo ""
echo "3. 或分步执行:"
echo "   python src/dataset.py        # 生成数据"
echo "   python src/train.py          # 训练模型"
echo "   python src/export_onnx.py    # 导出ONNX"
echo "   cd build && cmake .. && make # 编译"
echo "   ./build/inference models/trajectory_planner.onnx  # 推理"
echo ""
echo "注意: 每次打开新终端需要先激活虚拟环境"
echo "      source venv/bin/activate"
echo ""
