# 快速参考卡片

## 常用命令

### 环境配置

```bash
# 首次安装 (Ubuntu 24.04)
bash scripts/setup_ubuntu.sh

# 激活虚拟环境
source scripts/activate.sh
# 或
source venv/bin/activate

# 退出虚拟环境
deactivate
```

### 完整流程

```bash
# 一键运行 (虚拟环境版)
bash scripts/run_with_venv.sh
```

### 分步执行

```bash
# 1. 生成数据
python src/dataset.py

# 2. 训练模型
python src/train.py --model_type improved --epochs 50

# 3. 导出ONNX
python src/export_onnx.py

# 4. 编译
mkdir build && cd build && cmake .. && make

# 5. 推理
./build/inference models/trajectory_planner.onnx

# 6. 可视化
python visualize_cpp_output.py
```

### 测试和验证

```bash
# 测试模型一致性
python scripts/test_model.py --benchmark

# 快速测试 (使用预训练模型)
bash scripts/quick_start.sh
```

## 环境变量

### 虚拟环境激活后

```bash
# 检查Python路径
which python  # 应指向 venv/bin/python

# 检查已安装包
pip list | grep -E "torch|onnx"
```

### ONNX Runtime

```bash
# 设置路径
export ONNXRUNTIME_ROOT=/path/to/onnxruntime
export LD_LIBRARY_PATH=$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH

# 验证
ls $ONNXRUNTIME_ROOT/include/onnxruntime_cxx_api.h
ls $ONNXRUNTIME_ROOT/lib/libonnxruntime.so
```

## 添加到 ~/.bashrc

```bash
# 快捷激活别名
alias tpdemo='cd ~/workspace/llm/trajectory_planning_demo && source scripts/activate.sh'

# ONNX Runtime路径
export ONNXRUNTIME_ROOT=$HOME/path/to/onnxruntime
export LD_LIBRARY_PATH=$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH
```

使用:

```bash
source ~/.bashrc
tpdemo  # 快速激活并进入项目
```

## 目录结构速查

```
trajectory_planning_demo/
├── venv/              # 虚拟环境 (创建后)
├── data/              # 训练数据 *.pkl
├── models/            # 模型文件 *.pth *.onnx
├── build/             # C++编译产物
├── src/               # 源代码 *.py *.cpp
├── scripts/           # 脚本 *.sh *.py
└── include/           # C++头文件
```

## 常见路径

```bash
# Python解释器
venv/bin/python

# pip包管理
venv/bin/pip

# 可执行文件
build/inference

# 模型文件
models/best_model.pth
models/trajectory_planner.onnx

# 训练数据
data/train_data.pkl

# 配置文件
CMakeLists.txt
requirements.txt
```

## 故障排除速查

| 问题               | 命令                                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------------ |
| 虚拟环境未激活     | `source scripts/activate.sh`                                                                         |
| Python包缺失       | `pip install -r requirements.txt`                                                                    |
| ONNX Runtime找不到 | `export ONNXRUNTIME_ROOT=/path/to/onnxruntime`                                                       |
| 共享库找不到       | `export LD_LIBRARY_PATH=$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH`                                      |
| CMake失败          | `rm -rf build && mkdir build && cd build && cmake ..`                                                |
| 编译错误           | `sudo apt install build-essential cmake`                                                             |
| 虚拟环境损坏       | `rm -rf venv && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt` |

## Python虚拟环境命令

```bash
# 创建
python3 -m venv venv

# 激活
source venv/bin/activate

# 退出
deactivate

# 查看已安装包
pip list

# 安装包
pip install <package>

# 导出依赖
pip freeze > requirements.txt

# 安装依赖
pip install -r requirements.txt

# 升级pip
pip install --upgrade pip
```

## 系统包管理 (Ubuntu)

```bash
# 更新包列表
sudo apt update

# 安装构建工具
sudo apt install build-essential cmake

# 安装Python虚拟环境
sudo apt install python3-venv python3-dev

# 查看已安装包
dpkg -l | grep python3

# 搜索包
apt search python3
```

## 文件权限

```bash
# 添加执行权限
chmod +x scripts/*.sh

# 修复虚拟环境权限
chmod +x venv/bin/*
```

## 性能测试

```bash
# Python推理时间
time python -c "import onnxruntime as ort; ..."

# C++推理时间
time ./build/inference models/trajectory_planner.onnx

# 批量测试
python scripts/test_model.py --benchmark
```

## 清理

```bash
# 清理编译产物
rm -rf build/

# 清理数据
rm -rf data/*.pkl data/*.png

# 清理模型
rm -rf models/*.pth models/*.onnx models/*.png

# 完全清理 (保留源码)
rm -rf venv/ build/ data/*.pkl models/*.pth models/*.onnx

# 完全重置
git clean -fdx  # 如果使用git
```

## 网络资源

- PyTorch: https://pytorch.org/
- ONNX Runtime: https://onnxruntime.ai/
- Ubuntu文档: https://ubuntu.com/server/docs

## 获取帮助

```bash
# 查看README
cat README.md

# 查看安装指南
cat INSTALL.md

# 查看快速开始
cat QUICKSTART.md

# 查看脚本帮助
bash scripts/setup_ubuntu.sh --help 2>/dev/null || echo "无额外参数"
```
