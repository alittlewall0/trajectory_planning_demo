#!/bin/bash

# 激活项目虚拟环境
# 使用方法: source scripts/activate.sh

# 获取项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 激活虚拟环境
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"

    # 设置ONNX Runtime路径
    if [ -f "$PROJECT_ROOT/onnxruntime-linux-x64-1.16.0/lib/libonnxruntime.so" ]; then
        export ONNXRUNTIME_ROOT="$PROJECT_ROOT/onnxruntime-linux-x64-1.16.0"
        export LD_LIBRARY_PATH="$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH"
    fi

    # 设置项目根目录
    export PROJECT_ROOT="$PROJECT_ROOT"

    echo "✓ 虚拟环境已激活"
    echo "  Python: $(which python)"
    echo "  项目: $PROJECT_ROOT"
    if [ -n "$ONNXRUNTIME_ROOT" ]; then
        echo "  ONNX Runtime: $ONNXRUNTIME_ROOT"
    fi

    # 切换到项目目录
    cd "$PROJECT_ROOT"
else
    echo "错误: 虚拟环境不存在"
    echo "请先运行: bash scripts/setup_ubuntu.sh"
    return 1
fi
