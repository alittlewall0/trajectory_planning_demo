/**
 * 简化版C++推理示例
 * 最小化代码，便于理解ONNX Runtime的基本用法
 */

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

int main() {
    // 1. 创建ONNX Runtime环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    Ort::Session session{env, "models/trajectory_planner.onnx", session_options};

    // 2. 准备输入数据
    // 起点: (2, 2), 终点: (18, 18)
    std::vector<float> start_input = {2.0f, 2.0f};
    std::vector<float> goal_input = {18.0f, 18.0f};

    // 输入形状: [batch_size=1, features=2]
    std::vector<int64_t> input_shape = {1, 2};

    // 3. 创建输入tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );

    Ort::Value start_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        start_input.data(),
        start_input.size(),
        input_shape.data(),
        input_shape.size()
    );

    Ort::Value goal_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        goal_input.data(),
        goal_input.size(),
        input_shape.data(),
        input_shape.size()
    );

    // 4. 设置输入输出名称
    const char* input_names[] = {"start", "goal"};
    const char* output_names[] = {"trajectory"};

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(start_tensor));
    inputs.push_back(std::move(goal_tensor));

    // 5. 运行推理
    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        inputs.data(),
        inputs.size(),
        output_names,
        1
    );

    // 6. 获取结果
    float* output_data = outputs[0].GetTensorMutableData<float>();
    auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

    // 7. 打印轨迹点
    std::cout << "生成的轨迹点:\n";
    for (int i = 0; i < output_shape[1]; i++) {
        float x = output_data[i * 2];
        float y = output_data[i * 2 + 1];
        std::cout << "  点 " << i << ": (" << x << ", " << y << ")\n";
    }

    return 0;
}
