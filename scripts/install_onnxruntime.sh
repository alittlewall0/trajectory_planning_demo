#!/bin/bash

# 下载并安装ONNX Runtime预编译版本

VERSION="1.16.0"
ARCH="linux-x64"

echo "=========================================="
echo "  下载ONNX Runtime预编译版本"
echo "  版本: $VERSION"
echo "=========================================="
echo ""

URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/onnxruntime-${ARCH}-${VERSION}.tgz"

echo "下载地址: $URL"
echo ""

# 下载
if command -v wget &> /dev/null; then
    wget "$URL"
elif command -v curl &> /dev/null; then
    curl -L -O "$URL"
else
    echo "错误: 需要wget或curl"
    exit 1
fi

# 解压
echo "解压..."
tar -xzf onnxruntime-${ARCH}-${VERSION}.tgz

# 设置环境变量
INSTALL_DIR=$(pwd)/onnxruntime-${ARCH}-${VERSION}
export ONNXRUNTIME_ROOT=$INSTALL_DIR

echo ""
echo "=========================================="
echo "安装完成!"
echo "=========================================="
echo ""
echo "设置环境变量:"
echo "  export ONNXRUNTIME_ROOT=$INSTALL_DIR"
echo ""
echo "添加到~/.bashrc:"
echo "  echo 'export ONNXRUNTIME_ROOT=$INSTALL_DIR' >> ~/.bashrc"
echo ""
echo "文件结构:"
ls -lh $INSTALL_DIR/
echo ""

# 检查关键文件
if [ -f "$INSTALL_DIR/include/onnxruntime_cxx_api.h" ]; then
    echo "✓ C++头文件已就绪"
else
    echo "✗ C++头文件未找到"
fi

if [ -f "$INSTALL_DIR/lib/libonnxruntime.so" ]; then
    echo "✓ 库文件已就绪"
else
    echo "✗ 库文件未找到"
fi
