# Ubuntu 24.04 安装指南

## 系统要求

- Ubuntu 24.04
- Python 3.10+ (系统自带)
- 至少4GB可用内存
- 2GB可用磁盘空间

## 一键安装

```bash
cd trajectory_planning_demo
bash scripts/setup_ubuntu.sh
```

这将自动完成:
1. 安装系统依赖 (python3-venv, build-essential, cmake)
2. 创建Python虚拟环境
3. 安装Python依赖包
4. 下载并配置ONNX Runtime

## 详细安装步骤

### 1. 安装系统依赖

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential cmake git
```

### 2. 创建并激活虚拟环境

```bash
# 在项目目录下
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装Python依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

依赖包包括:
- torch (PyTorch)
- onnx (ONNX导出)
- onnxruntime (ONNX推理)
- numpy, matplotlib, tqdm

### 4. 安装ONNX Runtime (C++库)

#### 方法1: 下载预编译版本 (推荐)

```bash
bash scripts/install_onnxruntime.sh
```

这将下载 `onnxruntime-linux-x64-1.16.0` 到项目目录。

#### 方法2: 使用Conda

```bash
# 需要先安装conda
conda install -c conda-forge onnxruntime
export ONNXRUNTIME_ROOT=$CONDA_PREFIX
```

#### 方法3: 仅使用Python库

```bash
pip install onnxruntime
```

注意: 此方法只能用Python推理，C++编译需要额外的C++库。

### 5. 设置环境变量

创建 `~/.bashrc` 添加:

```bash
# 项目虚拟环境快速激活
alias tpdemo='cd ~/path/to/trajectory_planning_demo && source scripts/activate.sh'

# 或手动设置ONNX Runtime路径
export ONNXRUNTIME_ROOT=/path/to/onnxruntime
export LD_LIBRARY_PATH=$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH
```

## 使用

### 快速开始

```bash
# 方式1: 使用一键运行脚本
bash scripts/run_with_venv.sh

# 方式2: 手动激活后运行
source scripts/activate.sh
python src/dataset.py
python src/train.py --epochs 50
# ... 其他步骤
```

### 激活虚拟环境

```bash
# 使用激活脚本
source scripts/activate.sh

# 或直接激活
source venv/bin/activate
```

### 退出虚拟环境

```bash
deactivate
```

## 常见问题

### Q1: 虚拟环境创建失败

```bash
错误: Command '['/path/to/venv/bin/python3', '-Im', 'ensurepip', '--upgrade', 'default-pip']' returned non-zero exit status 1
```

**解决**:
```bash
sudo apt install --reinstall python3-venv
```

### Q2: PyTorch安装失败

```bash
错误: Could not find a version that satisfies the requirement torch
```

**解决**:
```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Q3: CMake找不到ONNX Runtime

```bash
错误: Could NOT find ONNXRUNTIME_INCLUDE_DIR
```

**解决**:
```bash
export ONNXRUNTIME_ROOT=~/path/to/trajectory_planning_demo/onnxruntime-linux-x64-1.16.0
cd build
cmake ..
```

或在 `CMakeLists.txt` 中手动设置:
```cmake
set(ONNXRUNTIME_ROOT "/absolute/path/to/onnxruntime")
```

### Q4: 运行时找不到共享库

```bash
错误: error while loading shared libraries: libonnxruntime.so
```

**解决**:
```bash
export LD_LIBRARY_PATH=$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH
```

或添加到 `~/.bashrc`:
```bash
echo 'export LD_LIBRARY_PATH=$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### Q5: 虚拟环境每次需要手动激活

**解决**: 在 `~/.bashrc` 添加别名:
```bash
alias tpdemo='cd ~/workspace/llm/trajectory_planning_demo && source scripts/activate.sh'
```

然后使用:
```bash
tpdemo  # 自动激活并进入项目目录
```

## 目录权限

如果遇到权限问题:

```bash
# 确保虚拟环境有执行权限
chmod +x venv/bin/*
chmod +x scripts/*.sh

# 或重新创建虚拟环境
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 卸载

```bash
# 删除虚拟环境
rm -rf venv

# 删除ONNX Runtime
rm -rf onnxruntime-linux-x64-*

# 删除构建文件
rm -rf build

# 删除数据
rm -rf data/*.pkl models/*.pth models/*.onnx
```

## 下一步

安装完成后，查看:
- [QUICKSTART.md](QUICKSTART.md) - 快速开始
- [README.md](README.md) - 完整文档

## 获取帮助

如果遇到问题:
1. 检查虚拟环境是否激活: `which python`
2. 检查ONNX Runtime路径: `echo $ONNXRUNTIME_ROOT`
3. 查看错误日志: 运行命令时添加 `-v` 或 `--verbose`
4. 检查系统依赖: `dpkg -l | grep python3-venv`
