"""
轨迹规划神经网络模型
基于MLP的序列预测模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryPlannerModel(nn.Module):
    """
    轨迹规划模型
    输入: 起点 [batch, 2], 终点 [batch, 2]
    输出: 轨迹点序列 [batch, seq_len, 2]
    """

    def __init__(self, hidden_dim=256, num_layers=4, trajectory_length=20):
        super(TrajectoryPlannerModel, self).__init__()

        self.trajectory_length = trajectory_length
        self.input_dim = 4  # start(2) + goal(2)
        self.output_dim = 2  # (x, y)

        # 输入嵌入层
        self.input_embedding = nn.Linear(self.input_dim, hidden_dim)

        # 主干网络 - 使用多层MLP
        self.backbone_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.backbone_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.backbone_layers.append(nn.ReLU())
            self.backbone_layers.append(nn.Dropout(0.1))

        self.backbone = nn.Sequential(*self.backbone_layers)

        # 输出层 - 为每个时间步预测坐标
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, self.output_dim)
            for _ in range(trajectory_length - 2)  # 减去起点和终点
        ])

        # 终点预测层（用于验证）
        self.goal_predictor = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, start, goal):
        """
        前向传播
        Args:
            start: [batch, 2] 起点坐标
            goal: [batch, 2] 终点坐标
        Returns:
            trajectory: [batch, trajectory_length, 2] 完整轨迹
        """
        batch_size = start.shape[0]

        # 拼接输入
        x = torch.cat([start, goal], dim=1)  # [batch, 4]

        # 输入嵌入
        x = self.input_embedding(x)  # [batch, hidden_dim]
        x = F.relu(x)

        # 主干网络
        features = self.backbone(x)  # [batch, hidden_dim]

        # 生成中间轨迹点
        trajectory_points = []
        for output_layer in self.output_layers:
            point = output_layer(features)  # [batch, 2]
            trajectory_points.append(point.unsqueeze(1))

        middle_points = torch.cat(trajectory_points, dim=1)  # [batch, traj_len-2, 2]

        # 组装完整轨迹: 起点 + 中间点 + 终点
        start_expanded = start.unsqueeze(1)  # [batch, 1, 2]
        goal_expanded = goal.unsqueeze(1)    # [batch, 1, 2]

        trajectory = torch.cat([
            start_expanded,
            middle_points,
            goal_expanded
        ], dim=1)  # [batch, trajectory_length, 2]

        return trajectory

    def predict_stepwise(self, start, goal):
        """
        逐步预测模式（用于推理）
        可以用于自回归生成
        """
        with torch.no_grad():
            return self.forward(start, goal)


class ImprovedTrajectoryModel(nn.Module):
    """
    改进的轨迹规划模型
    使用Transformer架构
    """

    def __init__(self, hidden_dim=128, num_heads=4, num_layers=3, trajectory_length=20):
        super(ImprovedTrajectoryModel, self).__init__()

        self.trajectory_length = trajectory_length
        self.hidden_dim = hidden_dim

        # 输入投影
        self.input_projection = nn.Linear(4, hidden_dim)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, trajectory_length, hidden_dim))

        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, 2)

    def forward(self, start, goal):
        """
        Args:
            start: [batch, 2]
            goal: [batch, 2]
        Returns:
            trajectory: [batch, trajectory_length, 2]
        """
        batch_size = start.shape[0]

        # 输入嵌入
        x = torch.cat([start, goal], dim=1)  # [batch, 4]
        x = self.input_projection(x)  # [batch, hidden_dim]
        x = x.unsqueeze(1)  # [batch, 1, hidden_dim]

        # 扩展到序列长度
        x = x.repeat(1, self.trajectory_length, 1)  # [batch, traj_len, hidden_dim]

        # 添加位置编码
        x = x + self.pos_embedding

        # Transformer编码
        x = self.transformer(x)  # [batch, traj_len, hidden_dim]

        # 输出投影
        trajectory = self.output_projection(x)  # [batch, traj_len, 2]

        # 强制第一个点和最后一个点
        trajectory[:, 0, :] = start
        trajectory[:, -1, :] = goal

        return trajectory


if __name__ == '__main__':
    # 测试模型
    batch_size = 4
    start = torch.randn(batch_size, 2)
    goal = torch.randn(batch_size, 2)

    # 测试基础模型
    print("测试基础模型...")
    model = TrajectoryPlannerModel(hidden_dim=256, num_layers=4, trajectory_length=20)
    trajectory = model(start, goal)
    print(f"输入: start {start.shape}, goal {goal.shape}")
    print(f"输出: trajectory {trajectory.shape}")

    # 测试改进模型
    print("\n测试改进模型...")
    improved_model = ImprovedTrajectoryModel(hidden_dim=128, num_heads=4, trajectory_length=20)
    trajectory = improved_model(start, goal)
    print(f"输出: trajectory {trajectory.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n基础模型参数量: {total_params:,}")

    total_params = sum(p.numel() for p in improved_model.parameters())
    print(f"改进模型参数量: {total_params:,}")
