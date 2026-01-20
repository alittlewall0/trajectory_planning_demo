"""
测试脚本 - 验证PyTorch和ONNX模型的一致性
"""
import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from model import TrajectoryPlannerModel, ImprovedTrajectoryModel


def test_pytorch_vs_onnx(pytorch_model_path, onnx_model_path, model_type='improved'):
    """对比PyTorch和ONNX的推理结果"""

    print("==========================================")
    print("  模型一致性测试")
    print "=========================================="
    print("")

    # 加载PyTorch模型
    print(f"加载PyTorch模型: {pytorch_model_path}")
    checkpoint = torch.load(pytorch_model_path, map_location='cpu')

    trajectory_length = 20  # 默认轨迹长度

    if model_type == 'basic':
        pytorch_model = TrajectoryPlannerModel(
            hidden_dim=128,
            num_layers=4,
            trajectory_length=trajectory_length
        )
    else:
        pytorch_model = ImprovedTrajectoryModel(
            hidden_dim=128,
            num_heads=4,
            num_layers=3,
            trajectory_length=trajectory_length
        )

    pytorch_model.load_state_dict(checkpoint['model_state_dict'])
    pytorch_model.eval()

    # 加载ONNX模型
    print(f"加载ONNX模型: {onnx_model_path}")
    ort_session = ort.InferenceSession(onnx_model_path)

    # 生成测试数据
    test_cases = [
        ((2.0, 2.0), (18.0, 18.0), "对角线"),
        ((5.0, 5.0), (15.0, 15.0), "短距离对角"),
        ((2.0, 10.0), (18.0, 10.0), "水平线"),
        ((10.0, 2.0), (10.0, 18.0), "垂直线"),
    ]

    print("\n测试结果:")
    print("-" * 80)

    max_diff = 0

    for (start, goal), description in [(tc[:2], tc[2]) for tc in test_cases]:
        # 准备输入
        start_np = np.array([[start[0], start[1]]], dtype=np.float32)
        goal_np = np.array([[goal[0], goal[1]]], dtype=np.float32)

        start_tensor = torch.FloatTensor(start_np)
        goal_tensor = torch.FloatTensor(goal_np)

        # PyTorch推理
        with torch.no_grad():
            pytorch_output = pytorch_model(start_tensor, goal_tensor).numpy()

        # ONNX推理
        onnx_output = ort_session.run(
            None,
            {
                'start': start_np,
                'goal': goal_np
            }
        )[0]

        # 计算差异
        diff = np.abs(pytorch_output - onnx_output)
        max_diff_case = diff.max()
        mean_diff_case = diff.mean()

        if max_diff_case > max_diff:
            max_diff = max_diff_case

        print(f"\n测试: {description}")
        print(f"  起点: {start}, 终点: {goal}")
        print(f"  最大差异: {max_diff_case:.8f}")
        print(f"  平均差异: {mean_diff_case:.8f}")

        if max_diff_case < 1e-5:
            print(f"  状态: ✓ 通过")
        elif max_diff_case < 1e-3:
            print(f"  状态: ⚠ 警告 (小差异)")
        else:
            print(f"  状态: ✗ 失败 (大差异)")

    print("\n" + "=" * 80)
    print(f"总体最大差异: {max_diff:.8f}")

    if max_diff < 1e-5:
        print("✓ 所有测试通过! PyTorch和ONNX模型输出一致")
        return True
    elif max_diff < 1e-3:
        print("⚠ 存在小差异，可能是数值精度问题")
        return True
    else:
        print("✗ 测试失败! 存在显著差异")
        return False


def benchmark_onnx(onnx_model_path, num_runs=100):
    """ONNX推理性能测试"""

    print("\n==========================================")
    print("  ONNX性能测试")
    print("==========================================")
    print("")

    ort_session = ort.InferenceSession(onnx_model_path)

    # 准备测试数据
    start = np.random.rand(1, 2).astype(np.float32) * 20
    goal = np.random.rand(1, 2).astype(np.float32) * 20

    # 预热
    for _ in range(10):
        _ = ort_session.run(None, {'start': start, 'goal': goal})

    # 测试
    import time
    times = []

    for _ in range(num_runs):
        start_time = time.perf_counter()
        _ = ort_session.run(None, {'start': start, 'goal': goal})
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # ms

    times = np.array(times)

    print(f"运行次数: {num_runs}")
    print(f"平均时间: {times.mean():.2f} ms")
    print(f"最小时间: {times.min():.2f} ms")
    print(f"最大时间: {times.max():.2f} ms")
    print(f"标准差: {times.std():.2f} ms")
    print(f"中位数: {np.median(times):.2f} ms")
    print(f"FPS: {1000 / times.mean():.1f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='测试和基准测试')
    parser.add_argument('--pytorch_model', type=str, default='models/best_model.pth',
                       help='PyTorch模型路径')
    parser.add_argument('--onnx_model', type=str, default='models/trajectory_planner.onnx',
                       help='ONNX模型路径')
    parser.add_argument('--model_type', type=str, default='improved',
                       choices=['basic', 'improved'])
    parser.add_argument('--benchmark', action='store_true',
                       help='运行性能测试')

    args = parser.parse_args()

    # 一致性测试
    success = test_pytorch_vs_onnx(
        args.pytorch_model,
        args.onnx_model,
        args.model_type
    )

    # 性能测试
    if args.benchmark:
        benchmark_onnx(args.onnx_model)

    exit(0 if success else 1)
