"""
轨迹规划数据集生成器
模拟A*搜索风格的轨迹数据
"""
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple


class TrajectoryDataset:
    """轨迹数据集类"""

    def __init__(self, grid_size=20, num_samples=1000, trajectory_length=20):
        self.grid_size = grid_size
        self.num_samples = num_samples
        self.trajectory_length = trajectory_length
        self.data = []

    def generate_astar_like_trajectory(self, start, goal) -> np.ndarray:
        """
        生成类似A*的轨迹（带有一定的随机性）
        返回: [trajectory_length, 2] 的轨迹点坐标
        """
        trajectory = [start]
        current = np.array(start, dtype=float)

        # 计算总距离
        total_distance = np.linalg.norm(np.array(goal) - np.array(start))

        for i in range(self.trajectory_length - 2):
            # 计算方向向量
            direction = np.array(goal) - current
            distance = np.linalg.norm(direction)

            if distance < 0.5:
                # 接近目标，添加小随机扰动
                noise = np.random.randn(2) * 0.3
                next_point = current + direction + noise
            else:
                # 沿着目标方向前进，添加随机探索
                normalized_dir = direction / (distance + 1e-6)
                progress = total_distance / (self.trajectory_length - i)

                # 添加随机性（模拟A*的探索）
                noise = np.random.randn(2) * 0.5
                next_point = current + normalized_dir * progress + noise

            # 限制在grid范围内
            next_point = np.clip(next_point, 0, self.grid_size - 1)
            trajectory.append(next_point.copy())
            current = next_point

        trajectory.append(goal)
        return np.array(trajectory, dtype=np.float32)

    def generate_dataset(self):
        """生成完整数据集"""
        print(f"正在生成 {self.num_samples} 条轨迹数据...")

        for i in range(self.num_samples):
            # 随机生成起点和终点
            start = np.random.rand(2) * self.grid_size
            goal = np.random.rand(2) * self.grid_size

            # 生成轨迹
            trajectory = self.generate_astar_like_trajectory(start, goal)

            self.data.append({
                'start': start.astype(np.float32),
                'goal': goal.astype(np.float32),
                'trajectory': trajectory
            })

            if (i + 1) % 100 == 0:
                print(f"  已生成 {i + 1}/{self.num_samples} 条轨迹")

        return self.data

    def save(self, path: str):
        """保存数据集"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"数据集已保存到: {path}")

    @staticmethod
    def load(path: str):
        """加载数据集"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

    def visualize_sample(self, idx=0, save_path=None):
        """可视化样本轨迹"""
        import matplotlib.pyplot as plt

        if idx >= len(self.data):
            idx = 0

        sample = self.data[idx]
        trajectory = sample['trajectory']

        plt.figure(figsize=(10, 10))
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-o',
                 linewidth=2, markersize=4, label='Trajectory')
        plt.plot(sample['start'][0], sample['start'][1],
                 'g*', markersize=20, label='Start')
        plt.plot(sample['goal'][0], sample['goal'][1],
                 'r*', markersize=20, label='Goal')

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(0, self.grid_size)
        plt.ylim(0, self.grid_size)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Trajectory Sample {idx}')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化已保存到: {save_path}")
        else:
            plt.show()


if __name__ == '__main__':
    # 生成数据集
    dataset = TrajectoryDataset(
        grid_size=20,
        num_samples=1000,
        trajectory_length=20
    )

    data = dataset.generate_dataset()
    dataset.save('data/train_data.pkl')

    # 可视化几个样本
    for i in range(3):
        dataset.visualize_sample(i, save_path=f'data/sample_{i}.png')
