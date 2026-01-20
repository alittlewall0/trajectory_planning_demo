# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a deep learning model inference pipeline demonstrating the complete ML lifecycle from training to production deployment. The project implements trajectory planning for autonomous systems using neural networks, taking start and goal coordinates as input and outputting a complete trajectory (similar to A* search but learned).

**Key Characteristics:**
- Multi-language project: Python for ML/training, C++ for production inference, Bash for automation
- Two-stage architecture: Training in Python, deployment in C++ via ONNX
- Ubuntu 24.04 optimized with Python virtual environment isolation
- Complete pipeline: data generation → training → ONNX export → C++ deployment

## Architecture

### Model Variants

The codebase includes two model architectures in [src/model.py](src/model.py):

1. **TrajectoryPlannerModel** (basic): Multi-layer perceptron with dropout
   - Input: start[2] + goal[2] → concatenated to [4]
   - Hidden: 256 dim, 4 layers with ReLU and Dropout(0.1)
   - Output: trajectory[20, 2] - 20 waypoints with (x, y) coordinates

2. **ImprovedTrajectoryModel** (improved): Transformer encoder architecture
   - Input: start[2] + goal[2] → concatenated to [4]
   - Hidden: 128 dim, 3 Transformer encoder layers, 4 heads
   - Features: Positional embedding, batch-first processing
   - Output: trajectory[20, 2] with forced start/goal endpoints

### Data Flow

```
Input: start(x1, y1) + goal(x2, y2)
  ↓
Neural Network (PyTorch)
  ↓
Training & Validation
  ↓
ONNX Export (Opset 17)
  ↓
C++ Inference (ONNX Runtime)
  ↓
Output: 20-point trajectory sequence [batch, 20, 2]
```

### Component Interaction

- **[src/dataset.py](src/dataset.py)**: Generates 1000 synthetic A*-style trajectory samples
- **[src/model.py](src/model.py)**: Defines both model architectures
- **[src/train.py](src/train.py)**: Training pipeline with MSE loss, Adam optimizer, LR scheduler
- **[src/export_onnx.py](src/export_onnx.py)**: PyTorch → ONNX conversion with validation
- **[src/inference.cpp](src/inference.cpp)**: Production C++ inference using ONNX Runtime C++ API

## Common Commands

### Environment Setup

**Always activate the virtual environment first before Python commands:**

```bash
source scripts/activate.sh  # Or: source venv/bin/activate
```

Verify activation:
```bash
which python  # Should show: /path/to/venv/bin/python
```

### One-Command Workflow

```bash
bash scripts/setup_ubuntu.sh      # First-time setup (install dependencies)
bash scripts/run_with_venv.sh     # Complete pipeline: data → train → export → build → inference
```

### Individual Pipeline Steps

```bash
# 1. Generate training data (1000 samples)
python src/dataset.py
# Output: data/train_data.pkl, data/sample_*.png

# 2. Train model
python src/train.py --model_type improved --epochs 50 --batch_size 32 --lr 0.001 --hidden_dim 128
# Output: models/best_model.pth, models/loss_curve.png, models/predictions_epoch_*.png

# 3. Export to ONNX
python src/export_onnx.py --model_path models/best_model.pth --model_type improved --onnx_path models/trajectory_planner.onnx
# Output: models/trajectory_planner.onnx

# 4. Build C++ inference
mkdir -p build && cd build && cmake .. && make -j$(nproc) && cd ..
# Output: build/inference executable

# 5. Run C++ inference
./build/inference models/trajectory_planner.onnx
# Output: trajectory_output.csv, visualize_cpp_output.py

# 6. Test model consistency (PyTorch vs ONNX)
python scripts/test_model.py
```

### Model Selection

Use `--model_type` flag to choose architecture:
- `--model_type basic`: MLP-based TrajectoryPlannerModel
- `--model_type improved`: Transformer-based ImprovedTrajectoryModel (recommended)

## Build System

### Python Build
- **Virtual environment**: `venv/` (isolated dependencies)
- **Package manager**: pip with [requirements.txt](requirements.txt)
- **Key dependencies**: torch>=2.0.0, onnx>=1.14.0, onnxruntime>=1.15.0, numpy, matplotlib, tqdm

### C++ Build
- **Build tool**: CMake 3.14+
- **Compiler**: C++17 standard
- **Build config**: [CMakeLists.txt](CMakeLists.txt)
- **External library**: ONNX Runtime 1.16.0 (precompiled, located in `onnxruntime-linux-x64-1.16.0/`)

**CMake searches ONNX Runtime in this order:**
1. `$ONNXRUNTIME_ROOT` environment variable (set it if not in default path)
2. Common paths: `/usr/local`, `/usr`, `/opt/onnxruntime`, `$CONDA_PREFIX`
3. Falls back with warning if not found

**Required environment variables (if ONNX Runtime not in default path):**
```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.16.0
export LD_LIBRARY_PATH=$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH
```

## Testing

**No formal testing framework** (no pytest, unittest). Uses custom validation:

- **[scripts/test_model.py](scripts/test_model.py)**: Validates PyTorch vs ONNX Runtime output consistency
  - Tests: Diagonal, horizontal, vertical trajectories
  - Tolerance: < 1e-5 numerical accuracy
  - Includes benchmarking mode

Run tests:
```bash
source scripts/activate.sh
python scripts/test_model.py
```

## Code Conventions

### Python
- PEP 8 compliant (mostly)
- Type hints used sparingly
- Extensive docstrings in **Chinese**
- Each script is standalone executable with `if __name__ == '__main__'` pattern
- Functions use snake_case, classes use PascalCase

### C++
- Modern C++17 features
- RAII for resource management
- Clear error messages with exception handling
- CamelCase for methods, snake_case for functions

## Important Patterns

1. **Virtual Environment First**: All Python scripts assume activated venv
2. **Two-Model Architecture**: Always specify `--model_type` when training/exporting
3. **ONNX-Centric Deployment**: Heavy use of ONNX for cross-language deployment
4. **Visualization-First Validation**: Extensive matplotlib visualizations for debugging
5. **Automation-First**: Extensive shell scripts in [scripts/](scripts/) for common workflows

## Documentation Language

**All existing documentation is in Chinese** (README.md, INSTALL.md, QUICKSTART.md, REFERENCE.md).
- Code comments are in Chinese
- Docstrings are in Chinese
- Consider this when making changes - maintain consistency or add English translations

## Performance Benchmarks

Reference: Intel i7-10750H

| Platform | Inference Time | Throughput |
|----------|---------------|------------|
| PyTorch (Python) | ~10-20 ms | ~83 FPS |
| ONNX Runtime (Python) | ~5-10 ms | ~166 FPS |
| ONNX Runtime (C++) | ~2-5 ms | ~333 FPS |

## Troubleshooting Common Issues

1. **Virtual environment not activated**: Commands fail with module import errors
   - Solution: `source scripts/activate.sh` and verify with `which python`

2. **ONNX Runtime not found during C++ build**: `onnxruntime_cxx_api.h: No such file or directory`
   - Solution: `bash scripts/install_onnxruntime.sh` and `export ONNXRUNTIME_ROOT=$(pwd)/onnxruntime-linux-x64-1.16.0`

3. **Runtime library loading error**: `error while loading shared libraries: libonnxruntime.so`
   - Solution: `export LD_LIBRARY_PATH=$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH`

4. **Model mismatch**: Training with one model type but exporting with another
   - Solution: Always use consistent `--model_type` flag across train and export steps
