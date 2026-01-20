"""
ONNX模型导出脚本
"""
import torch
import torch.onnx
from pathlib import Path
import sys
import argparse

sys.path.append(str(Path(__file__).parent))

from model import TrajectoryPlannerModel, ImprovedTrajectoryModel


def export_to_onnx(model, onnx_path, input_size, trajectory_length, grid_size=20):
    """
    导出模型到ONNX格式

    Args:
        model: PyTorch模型
        onnx_path: ONNX保存路径
        input_size: 输入batch size
        trajectory_length: 轨迹长度
        grid_size: 网格大小
    """
    model.eval()

    # 创建示例输入
    start = torch.randn(input_size, 2)
    goal = torch.randn(input_size, 2)

    print(f"导出模型到: {onnx_path}")
    print(f"输入形状: start {start.shape}, goal {goal.shape}")

    try:
        torch.onnx.export(
            model,
            (start, goal),
            onnx_path,
            export_params=True,
            opset_version=17,  # 使用较新的opset版本
            do_constant_folding=True,
            input_names=['start', 'goal'],
            output_names=['trajectory'],
            dynamic_axes={
                'start': {0: 'batch_size'},
                'goal': {0: 'batch_size'},
                'trajectory': {0: 'batch_size'}
            }
        )
        print(f"✓ 模型导出成功!")
    except Exception as e:
        print(f"✗ 导出失败: {e}")
        return False

    # 验证ONNX模型
    print("\n验证ONNX模型...")
    try:
        import onnx
        import onnxruntime as ort

        # 加载并检查模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX模型检查通过")

        # 打印模型信息
        print(f"\n模型输入:")
        for input in onnx_model.graph.input:
            print(f"  - {input.name}: {input.type}")

        print(f"\n模型输出:")
        for output in onnx_model.graph.output:
            print(f"  - {output.name}: {output.type}")

        # 测试ONNX Runtime推理
        print("\n测试ONNX Runtime推理...")
        ort_session = ort.InferenceSession(onnx_path)

        # 准备输入
        start_np = start.numpy()
        goal_np = goal.numpy()

        # 推理
        outputs = ort_session.run(
            None,
            {
                'start': start_np,
                'goal': goal_np
            }
        )

        trajectory_onnx = outputs[0]

        # 与PyTorch输出对比
        with torch.no_grad():
            trajectory_torch = model(start, goal).numpy()

        diff = np.abs(trajectory_onnx - trajectory_torch).max()
        print(f"ONNX vs PyTorch 最大差异: {diff:.8f}")

        if diff < 1e-5:
            print("✓ ONNX模型验证成功!")
        else:
            print(f"⚠ 警告: 差异较大 ({diff})")

        return True

    except Exception as e:
        print(f"✗ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='导出模型到ONNX格式')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                       help='PyTorch模型路径')
    parser.add_argument('--onnx_path', type=str, default='models/trajectory_planner.onnx',
                       help='ONNX模型保存路径')
    parser.add_argument('--model_type', type=str, default='improved',
                       choices=['basic', 'improved'], help='模型类型')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--trajectory_length', type=int, default=20, help='轨迹长度')
    parser.add_argument('--grid_size', type=int, default=20, help='网格大小')
    parser.add_argument('--batch_size', type=int, default=1, help='导出的batch size')

    args = parser.parse_args()

    # 创建输出目录
    Path(args.onnx_path).parent.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"加载模型: {args.model_path}")

    checkpoint = torch.load(args.model_path, map_location='cpu')

    # 创建模型
    if args.model_type == 'basic':
        model = TrajectoryPlannerModel(
            hidden_dim=args.hidden_dim,
            num_layers=4,
            trajectory_length=args.trajectory_length
        )
    else:
        model = ImprovedTrajectoryModel(
            hidden_dim=args.hidden_dim,
            num_heads=4,
            num_layers=3,
            trajectory_length=args.trajectory_length
        )

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ 模型加载成功")
    print(f"  - Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  - Train Loss: {checkpoint.get('train_loss', 'unknown'):.6f}")
    print(f"  - Val Loss: {checkpoint.get('val_loss', 'unknown'):.6f}")

    # 导出ONNX
    success = export_to_onnx(
        model,
        args.onnx_path,
        args.batch_size,
        args.trajectory_length,
        args.grid_size
    )

    if success:
        print(f"\n✓ 导出完成: {args.onnx_path}")
        print(f"\n可以使用以下命令进行C++推理:")
        print(f"  ./build/inference {args.onnx_path}")
    else:
        print(f"\n✗ 导出失败")
        return 1

    return 0


if __name__ == '__main__':
    import numpy as np
    exit(main())
