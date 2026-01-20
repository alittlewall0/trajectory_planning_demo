"""
训练脚本
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent))

from model import TrajectoryPlannerModel, ImprovedTrajectoryModel
from dataset import TrajectoryDataset


class TrajectoryDataLoader(Dataset):
    """PyTorch数据集包装器"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        start = sample['start']
        goal = sample['goal']
        trajectory = sample['trajectory']

        # 转换为tensor
        start = torch.FloatTensor(start)
        goal = torch.FloatTensor(goal)
        trajectory = torch.FloatTensor(trajectory)

        return start, goal, trajectory


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc='Training')
    for start, goal, trajectory_gt in pbar:
        start = start.to(device)
        goal = goal.to(device)
        trajectory_gt = trajectory_gt.to(device)

        # 前向传播
        optimizer.zero_grad()
        trajectory_pred = model(start, goal)

        # 计算损失
        loss = criterion(trajectory_pred, trajectory_gt)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for start, goal, trajectory_gt in dataloader:
            start = start.to(device)
            goal = goal.to(device)
            trajectory_gt = trajectory_gt.to(device)

            trajectory_pred = model(start, goal)
            loss = criterion(trajectory_pred, trajectory_gt)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def visualize_predictions(model, dataloader, device, save_path, num_samples=4):
    """可视化预测结果"""
    model.eval()

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    with torch.no_grad():
        for i in range(num_samples):
            start, goal, trajectory_gt = dataloader.dataset[i]

            start_tensor = start.unsqueeze(0).to(device)
            goal_tensor = goal.unsqueeze(0).to(device)

            trajectory_pred = model(start_tensor, goal_tensor)[0].cpu().numpy()

            ax = axes[i]

            # 绘制真实轨迹
            ax.plot(trajectory_gt[:, 0], trajectory_gt[:, 1],
                   'g-o', linewidth=2, markersize=4, label='Ground Truth', alpha=0.7)

            # 绘制预测轨迹
            ax.plot(trajectory_pred[:, 0], trajectory_pred[:, 1],
                   'r--s', linewidth=2, markersize=4, label='Prediction', alpha=0.7)

            # 标记起点和终点
            ax.plot(start[0], start[1], 'go', markersize=15, label='Start')
            ax.plot(goal[0], goal[1], 'ro', markersize=15, label='Goal')

            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'Sample {i+1}')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"预测可视化已保存到: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='训练轨迹规划模型')
    parser.add_argument('--data_path', type=str, default='data/train_data.pkl',
                       help='训练数据路径')
    parser.add_argument('--model_type', type=str, default='improved',
                       choices=['basic', 'improved'], help='模型类型')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='模型保存目录')
    parser.add_argument('--grid_size', type=int, default=20, help='网格大小')

    args = parser.parse_args()

    # 创建保存目录
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    print(f"加载数据: {args.data_path}")
    data = TrajectoryDataset.load(args.data_path)

    # 划分训练集和验证集
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_data, val_data = data[:train_size], data[train_size:]

    print(f"训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}")

    # 创建数据加载器
    train_dataset = TrajectoryDataLoader(train_data)
    val_dataset = TrajectoryDataLoader(val_data)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)

    # 创建模型
    trajectory_length = len(data[0]['trajectory'])
    if args.model_type == 'basic':
        model = TrajectoryPlannerModel(
            hidden_dim=args.hidden_dim,
            num_layers=4,
            trajectory_length=trajectory_length
        ).to(device)
    else:
        model = ImprovedTrajectoryModel(
            hidden_dim=args.hidden_dim,
            num_heads=4,
            num_layers=3,
            trajectory_length=trajectory_length
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print(f"\n开始训练 ({args.epochs} epochs)...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # 学习率调度
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = Path(args.save_dir) / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, save_path)
            print(f"保存最佳模型: {save_path}")

        # 定期可视化
        if (epoch + 1) % 20 == 0:
            visualize_predictions(
                model, val_loader, device,
                f'{args.save_dir}/predictions_epoch_{epoch+1}.png'
            )

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{args.save_dir}/loss_curve.png', dpi=150)
    print(f"\n损失曲线已保存到: {args.save_dir}/loss_curve.png")

    # 最终可视化
    visualize_predictions(model, val_loader, device,
                         f'{args.save_dir}/final_predictions.png')

    print(f"\n训练完成!")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"模型保存在: {args.save_dir}/best_model.pth")


if __name__ == '__main__':
    main()
