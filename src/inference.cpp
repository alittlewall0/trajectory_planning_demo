/**
 * ONNX Runtime C++ 推理示例
 * 实现轨迹规划的模型推理
 */

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <fstream>

#include <onnxruntime_cxx_api.h>

// 轨迹结构体
struct TrajectoryPoint {
    float x, y;
};

struct Trajectory {
    std::vector<TrajectoryPoint> points;
    TrajectoryPoint start;
    TrajectoryPoint goal;

    void print() const {
        std::cout << "Start: (" << start.x << ", " << start.y << ")\n";
        std::cout << "Goal: (" << goal.x << ", " << goal.y << ")\n";
        std::cout << "Trajectory points (" << points.size() << "):\n";
        for (size_t i = 0; i < points.size(); ++i) {
            std::cout << "  [" << i << "]: (" << points[i].x << ", " << points[i].y << ")\n";
        }
    }

    void saveToFile(const std::string& filename) const {
        std::ofstream file(filename);
        if (file.is_open()) {
            file << start.x << "," << start.y << "\n";
            for (const auto& point : points) {
                file << point.x << "," << point.y << "\n";
            }
            file << goal.x << "," << goal.y << "\n";
            file.close();
            std::cout << "轨迹已保存到: " << filename << std::endl;
        }
    }
};

class TrajectoryPredictor {
private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;

    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;

    std::vector<int64_t> input_shape_start_;
    std::vector<int64_t> input_shape_goal_;
    std::vector<int64_t> output_shape_;

public:
    TrajectoryPredictor(const std::string& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "TrajectoryPredictor"),
          session_options_(),
          session_(env_, model_path.c_str(), session_options_),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

        // 设置输入输出名称
        input_names_ = {"start", "goal"};
        output_names_ = {"trajectory"};

        // 获取输入形状
        auto input_start = session_.GetInputTypeInfo(0);
        auto input_start_tensor = input_start.GetTensorTypeAndShapeInfo();
        input_shape_start_ = input_start_tensor.GetShape();

        auto input_goal = session_.GetInputTypeInfo(1);
        auto input_goal_tensor = input_goal.GetTensorTypeAndShapeInfo();
        input_shape_goal_ = input_shape_start_; // 假设形状相同

        // 获取输出形状
        auto output = session_.GetOutputTypeInfo(0);
        auto output_tensor = output.GetTensorTypeAndShapeInfo();
        output_shape_ = output_tensor.GetShape();

        std::cout << "模型加载成功!\n";
        std::cout << "输入 'start' 形状: [";
        for (auto dim : input_shape_start_) std::cout << dim << " ";
        std::cout << "]\n";
        std::cout << "输入 'goal' 形状: [";
        for (auto dim : input_shape_goal_) std::cout << dim << " ";
        std::cout << "]\n";
        std::cout << "输出 'trajectory' 形状: [";
        for (auto dim : output_shape_) std::cout << dim << " ";
        std::cout << "]\n";
    }

    Trajectory predict(float start_x, float start_y, float goal_x, float goal_y) {
        // 准备输入数据
        std::vector<float> start_input = {start_x, start_y};
        std::vector<float> goal_input = {goal_x, goal_y};

        // 创建输入形状 - 使用实际的batch size (1) 替换动态维度
        std::vector<int64_t> start_shape = {1, 2};
        std::vector<int64_t> goal_shape = {1, 2};

        // 创建输入tensor
        auto start_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            start_input.data(),
            start_input.size(),
            start_shape.data(),
            start_shape.size()
        );

        auto goal_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            goal_input.data(),
            goal_input.size(),
            goal_shape.data(),
            goal_shape.size()
        );

        // 执行推理
        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(start_tensor));
        inputs.push_back(std::move(goal_tensor));

        auto outputs = session_.Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            inputs.data(),
            inputs.size(),
            output_names_.data(),
            output_names_.size()
        );

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "推理时间: " << duration.count() << " ms\n";

        // 解析输出
        float* output_data = outputs[0].GetTensorMutableData<float>();
        auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

        Trajectory trajectory;
        trajectory.start = {start_x, start_y};
        trajectory.goal = {goal_x, goal_y};

        // 提取轨迹点
        int batch_size = output_shape[0];
        int trajectory_length = output_shape[1];

        for (int i = 0; i < trajectory_length; ++i) {
            int idx = i * 2; // 每个点有2个坐标
            trajectory.points.push_back({output_data[idx], output_data[idx + 1]});
        }

        return trajectory;
    }

    // 批量预测
    std::vector<Trajectory> predictBatch(const std::vector<std::pair<float, float>>& starts,
                                         const std::vector<std::pair<float, float>>& goals) {
        if (starts.size() != goals.size()) {
            throw std::runtime_error("Starts and goals must have the same size");
        }

        int batch_size = starts.size();

        // 准备批量输入
        std::vector<float> start_input;
        std::vector<float> goal_input;

        for (const auto& s : starts) {
            start_input.push_back(s.first);
            start_input.push_back(s.second);
        }

        for (const auto& g : goals) {
            goal_input.push_back(g.first);
            goal_input.push_back(g.second);
        }

        // 更新输入形状
        std::vector<int64_t> batch_input_shape = {batch_size, 2};

        // 创建输入tensor
        auto start_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            start_input.data(),
            start_input.size(),
            batch_input_shape.data(),
            batch_input_shape.size()
        );

        auto goal_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            goal_input.data(),
            goal_input.size(),
            batch_input_shape.data(),
            batch_input_shape.size()
        );

        // 执行推理
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(start_tensor));
        inputs.push_back(std::move(goal_tensor));

        auto outputs = session_.Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            inputs.data(),
            inputs.size(),
            output_names_.data(),
            output_names_.size()
        );

        // 解析输出
        float* output_data = outputs[0].GetTensorMutableData<float>();
        auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

        int trajectory_length = output_shape[1];

        std::vector<Trajectory> trajectories;
        for (int b = 0; b < batch_size; ++b) {
            Trajectory traj;
            traj.start = {starts[b].first, starts[b].second};
            traj.goal = {goals[b].first, goals[b].second};

            for (int i = 0; i < trajectory_length; ++i) {
                int idx = b * trajectory_length * 2 + i * 2;
                traj.points.push_back({output_data[idx], output_data[idx + 1]});
            }

            trajectories.push_back(traj);
        }

        return trajectories;
    }
};

// 生成Python可视化脚本
void generateVisualizationScript(const std::string& data_file) {
    std::ofstream script("visualize_cpp_output.py");
    script << R"(import matplotlib.pyplot as plt
import numpy as np

# 读取轨迹数据
data = []
with open(')" << data_file << R"(', 'r') as f:
    for line in f:
        x, y = map(float, line.strip().split(','))
        data.append([x, y])

data = np.array(data)

# 绘制轨迹
plt.figure(figsize=(10, 10))
plt.plot(data[1:-1, 0], data[1:-1, 1], 'b-o', linewidth=2, markersize=4, label='Trajectory')
plt.plot(data[0, 0], data[0, 1], 'g*', markersize=20, label='Start')
plt.plot(data[-1, 0], data[-1, 1], 'r*', markersize=20, label='Goal')

plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('C++ ONNX Runtime Inference Result')

plt.savefig('cpp_inference_result.png', dpi=150, bbox_inches='tight')
print("可视化已保存到: cpp_inference_result.png")
plt.show()
)";
    script.close();
    std::cout << "可视化脚本已生成: visualize_cpp_output.py\n";
}

int main(int argc, char* argv[]) {
    std::cout << "==========================================\n";
    std::cout << "  ONNX Runtime C++ 推理示例\n";
    std::cout << "  轨迹规划模型\n";
    std::cout << "==========================================\n\n";

    // 检查命令行参数
    std::string model_path;
    if (argc < 2) {
        model_path = "models/trajectory_planner.onnx";
        std::cout << "使用默认模型路径: " << model_path << "\n";
        std::cout << "用法: " << argv[0] << " <model_path>\n\n";
    } else {
        model_path = argv[1];
    }

    try {
        // 创建预测器
        TrajectoryPredictor predictor(model_path);

        std::cout << "\n==========================================\n";
        std::cout << "示例 1: 单次预测\n";
        std::cout << "==========================================\n\n";

        // 示例1: 从(2, 2)到(18, 18)
        float start_x = 2.0f, start_y = 2.0f;
        float goal_x = 18.0f, goal_y = 18.0f;

        std::cout << "输入:\n";
        std::cout << "  起点: (" << start_x << ", " << start_y << ")\n";
        std::cout << "  终点: (" << goal_x << ", " << goal_y << ")\n\n";

        Trajectory trajectory = predictor.predict(start_x, start_y, goal_x, goal_y);

        std::cout << "\n输出:\n";
        trajectory.print();

        // 保存轨迹到文件
        trajectory.saveToFile("trajectory_output.csv");

        // 生成可视化脚本
        generateVisualizationScript("trajectory_output.csv");

        std::cout << "\n==========================================\n";
        std::cout << "示例 2: 多个测试案例\n";
        std::cout << "==========================================\n\n";

        // 示例2: 多个测试案例 - 使用循环单次预测以避免动态batch问题
        std::vector<std::pair<float, float>> starts = {
            {5.0f, 5.0f},
            {15.0f, 5.0f},
            {5.0f, 15.0f}
        };

        std::vector<std::pair<float, float>> goals = {
            {15.0f, 15.0f},
            {5.0f, 15.0f},
            {15.0f, 5.0f}
        };

        std::cout << "推理 " << starts.size() << " 个案例...\n\n";

        std::vector<Trajectory> trajectories;
        for (size_t i = 0; i < starts.size(); ++i) {
            std::cout << "案例 " << (i + 1) << ": ";
            std::cout << "从 (" << starts[i].first << ", " << starts[i].second << ") ";
            std::cout << "到 (" << goals[i].first << ", " << goals[i].second << ")\n";

            Trajectory traj = predictor.predict(starts[i].first, starts[i].second,
                                                goals[i].first, goals[i].second);
            trajectories.push_back(traj);
            std::cout << "  完成! 轨迹点数: " << traj.points.size() << "\n\n";
        }

        for (size_t i = 0; i < trajectories.size(); ++i) {
            std::cout << "案例 " << i + 1 << ":\n";
            std::cout << "  起点: (" << trajectories[i].start.x << ", " << trajectories[i].start.y << ") -> ";
            std::cout << "终点: (" << trajectories[i].goal.x << ", " << trajectories[i].goal.y << ")\n";
            std::cout << "  轨迹点数: " << trajectories[i].points.size() << "\n\n";
        }

        std::cout << "\n==========================================\n";
        std::cout << "推理完成!\n";
        std::cout << "==========================================\n";

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
