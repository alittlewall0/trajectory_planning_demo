#!/bin/bash

# ONNX Runtime 安装脚本

echo "=========================================="
echo "  ONNX Runtime 安装脚本"
echo "=========================================="
echo ""

# 检测Python环境
if command -v conda &> /dev/null; then
    echo "检测到Conda环境"
    INSTALL_METHOD="conda"
elif command -v pip &> /dev/null; then
    echo "检测到Pip环境"
    INSTALL_METHOD="pip"
else
    echo "错误: 未找到Conda或Pip"
    exit 1
fi

echo ""
echo "选择安装方法:"
echo "1. Conda (推荐，包含C++库)"
echo "2. Pip (仅Python库，需要单独下载C++库)"
echo "3. 跳过安装"
read -p "请选择 [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "使用Conda安装ONNX Runtime..."
        conda install -c conda-forge onnxruntime -y

        # 设置环境变量
        CONDA_PREFIX=$(conda info --base)
        if [ -z "$CONDA_PREFIX" ]; then
            CONDA_PREFIX=$CONDA_PREFIX
        fi

        echo ""
        echo "设置环境变量..."
        export ONNXRUNTIME_ROOT=$CONDA_PREFIX

        # 检查安装
        if [ -f "$CONDA_PREFIX/include/onnxruntime_cxx_api.h" ]; then
            echo "✓ ONNX Runtime安装成功!"
            echo "  C++头文件: $CONDA_PREFIX/include/onnxruntime_cxx_api.h"
            echo "  库文件: $CONDA_PREFIX/lib/libonnxruntime.so"
        else
            echo "✗ C++库未找到，请手动安装"
        fi
        ;;

    2)
        echo ""
        echo "使用Pip安装ONNX Runtime..."
        pip install onnxruntime

        echo ""
        echo "注意: Pip版本通常不包含C++库"
        echo "请从以下地址下载完整版本:"
        echo "https://github.com/microsoft/onnxruntime/releases"
        ;;

    3)
        echo "跳过安装"
        exit 0
        ;;

    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "建议: 将以下命令添加到 ~/.bashrc"
echo "export ONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT"
echo "=========================================="
