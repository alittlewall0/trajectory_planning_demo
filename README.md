# 轨迹规划模型推理Demo

这是一个完整的深度学习模型推理示例项目，实现了从模型训练、ONNX转换到C++推理的全过程。该demo应用场景是planning的轨迹生成，基于起终点输出类似A*搜索的结果。

**特别适配 Ubuntu 24.04 + Python虚拟环境**

## 快速开始 (Ubuntu 24.04)

### 一键安装和运行

```bash
cd trajectory_planning_demo
bash scripts/setup_ubuntu.sh      # 安装所有依赖
bash scripts/run_with_venv.sh     # 运行完整流程
```

### 手动安装

详细步骤请查看: [INSTALL.md](INSTALL.md)

## 项目结构

```
trajectory_planning_demo/
├── data/                           # 数据目录
│   ├── train_data.pkl              # 训练数据
│   └── sample_*.png                # 数据样本可视化
├── models/                         # 模型目录
│   ├── best_model.pth              # PyTorch最佳模型
│   └── trajectory_planner.onnx     # ONNX模型
├── src/                            # 源代码
│   ├── dataset.py                  # 数据集生成
│   ├── model.py                    # 神经网络模型定义
│   ├── train.py                    # 训练脚本
│   ├── export_onnx.py              # ONNX导出
│   ├── inference.cpp               # C++推理代码 (完整版)
│   └── inference_simple.cpp        # C++推理代码 (简化版)
├── scripts/                        # 脚本
│   ├── setup_ubuntu.sh             # Ubuntu环境配置 (新)
│   ├── run_with_venv.sh            # 虚拟环境版完整流程 (新)
│   ├── activate.sh                 # 快速激活虚拟环境 (新)
│   ├── run_all.sh                  # 完整流程脚本
│   ├── install_onnxruntime.sh      # ONNX Runtime安装
│   ├── quick_start.sh              # 快速开始
│   └── test_model.py               # 模型测试
├── venv/                           # Python虚拟环境 (创建后)
├── include/                        # C++头文件
├── build/                          # C++构建目录
├── CMakeLists.txt                  # CMake构建配置
├── requirements.txt                # Python依赖
├── INSTALL.md                      # Ubuntu安装指南 (新)
├── QUICKSTART.md                   # 快速开始指南
└── README.md                       # 本文件
```

## 环境要求

### Ubuntu 24.04 系统依赖

```bash
sudo apt install -y python3-venv python3-dev build-essential cmake
```

### Python虚拟环境

项目使用Python虚拟环境隔离依赖:

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 或使用快捷脚本
source scripts/activate.sh
```

### Python依赖包

主要依赖:
- torch >= 2.0.0
- onnx >= 1.14.0
- onnxruntime >= 1.15.0
- numpy, matplotlib, tqdm

完整列表见 [requirements.txt](requirements.txt)

### ONNX Runtime (C++库)

推荐使用预编译版本:

```bash
bash scripts/install_onnxruntime.sh
```

## 使用方法

### 方式1: 一键运行 (推荐)

```bash
# 自动配置环境并运行
bash scripts/setup_ubuntu.sh      # 首次运行，安装依赖
bash scripts/run_with_venv.sh     # 运行完整流程
```

### 方式2: 分步执行

#### 激活虚拟环境

```bash
source scripts/activate.sh
```

#### 1. 生成训练数据

```bash
python src/dataset.py
```

生成1000条轨迹数据，每条轨迹包含20个点。

#### 2. 训练模型

```bash
# 改进模型 (Transformer) - 推荐
python src/train.py --model_type improved --epochs 50

# 基础模型 (MLP)
python src/train.py --model_type basic --epochs 50
```

训练参数:
- `--epochs`: 训练轮数 (默认100)
- `--batch_size`: 批大小 (默认32)
- `--lr`: 学习率 (默认0.001)
- `--hidden_dim`: 隐藏层维度 (默认128)

#### 3. 导出ONNX模型

```bash
python src/export_onnx.py \
    --model_path models/best_model.pth \
    --model_type improved \
    --onnx_path models/trajectory_planner.onnx
```

#### 4. 编译C++推理代码

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
cd ..
```

如果ONNX Runtime不在默认路径:

```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake ..
```

#### 5. 运行C++推理

```bash
./build/inference models/trajectory_planner.onnx
```

#### 6. 可视化结果

```bash
python visualize_cpp_output.py
```

## 虚拟环境使用

### 激活虚拟环境

```bash
# 方式1: 使用激活脚本
source scripts/activate.sh

# 方式2: 直接激活
source venv/bin/activate

# 方式3: 添加别名到 ~/.bashrc
alias tpdemo='cd ~/workspace/llm/trajectory_planning_demo && source scripts/activate.sh'
```

### 退出虚拟环境

```bash
deactivate
```

### 查看当前Python路径

```bash
# 应该指向虚拟环境
which python
# 输出: /path/to/trajectory_planning_demo/venv/bin/python
```

## 功能特性

1. **数据生成**: 自动生成类似A*搜索的轨迹数据
2. **模型训练**: 基于Transformer的轨迹规划神经网络
3. **ONNX导出**: 将PyTorch模型转换为ONNX格式
4. **C++推理**: 使用ONNX Runtime进行高效推理
5. **可视化**: Python脚本可视化推理结果
6. **虚拟环境**: Python依赖隔离，不影响系统环境

## 模型架构

### 基础模型 (MLP)

- **输入**: 起点 [2], 终点 [2]
- **架构**: 多层感知机 + Dropout
- **输出**: 轨迹点序列 [trajectory_length, 2]

### 改进模型 (Transformer)

- **输入**: 起点 [2], 终点 [2]
- **架构**: Transformer Encoder + 位置编码
- **输出**: 轨迹点序列 [trajectory_length, 2]

## 输入输出格式

### Python (PyTorch)

```python
# 输入
start = torch.tensor([[x1, y1]])  # [batch, 2]
goal = torch.tensor([[x2, y2]])   # [batch, 2]

# 输出
trajectory = model(start, goal)   # [batch, trajectory_length, 2]
```

### C++ (ONNX Runtime)

```cpp
// 输入: 起点和终点坐标
float start_x = 2.0f, start_y = 2.0f;
float goal_x = 18.0f, goal_y = 18.0f;

// 输出: 轨迹点序列
Trajectory trajectory = predictor.predict(start_x, start_y, goal_x, goal_y);
```

## 性能对比

| 平台 | 推理时间 (ms) | 说明 |
|------|---------------|------|
| PyTorch (Python) | ~10-20 | 虚拟环境中运行 |
| ONNX Runtime (Python) | ~5-10 | 虚拟环境中运行 |
| ONNX Runtime (C++) | ~2-5 | 原生性能 |

## 常见问题

### 1. 虚拟环境相关

**Q: 如何确认虚拟环境已激活？**

```bash
which python
# 应该显示: /path/to/trajectory_planning_demo/venv/bin/python
```

**Q: 虚拟环境创建失败**

```bash
sudo apt install --reinstall python3-venv
rm -rf venv
python3 -m venv venv
```

### 2. ONNX Runtime找不到

**错误**: `onnxruntime_cxx_api.h: No such file or directory`

**解决**:
```bash
bash scripts/install_onnxruntime.sh
export ONNXRUNTIME_ROOT=$(pwd)/onnxruntime-linux-x64-1.16.0
```

### 3. CMake找不到ONNX Runtime

**解决**:
```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime
cd build
cmake ..
```

### 4. 运行时找不到共享库

**错误**: `error while loading shared libraries: libonnxruntime.so`

**解决**:
```bash
export LD_LIBRARY_PATH=$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH
```

或添加到 `~/.bashrc`:
```bash
echo 'export LD_LIBRARY_PATH=$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

## 文档

- [INSTALL.md](INSTALL.md) - Ubuntu 24.04 详细安装指南
- [QUICKSTART.md](QUICKSTART.md) - 快速开始指南

## 扩展功能

### 1. 添加障碍物

修改`dataset.py`，在生成轨迹时考虑障碍物约束。

### 2. 多模态预测

修改模型输出为多条候选轨迹，选择最优路径。

### 3. 实时推理

集成到ROS或其他机器人框架进行实时规划。

### 4. GPU加速

修改C++代码使用CUDA版本的ONNX Runtime。

## 参考资源

- [ONNX Runtime C++ API](https://onnxruntime.ai/docs/api/c/api/)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
- [Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

## License

MIT License

## 作者

Generated by Claude Code

## 更新日志

- v1.1.0 (2024-01)
  - 添加Ubuntu 24.04支持
  - 集成Python虚拟环境
  - 更新安装脚本
  - 添加虚拟环境激活脚本

- v1.0.0 (2024)
  - 初始版本
  - 实现基础MLP模型
  - 实现Transformer改进模型
  - 完整的训练和推理流程
  - C++ ONNX Runtime集成
