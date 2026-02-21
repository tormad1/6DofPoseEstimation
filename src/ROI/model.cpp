#include <onnxruntime_cxx_api.h>
#include <iostream>


void test_loadModel() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YoloCropper");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session session(env, L"resources/models/yolov8m.onnx", sessionOptions);
    std::cout << "Model loaded.\n";
}

