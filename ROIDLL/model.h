#pragma once
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

// Detection Decleration
struct Detection {
    cv::Rect2f box;   // in original image coordinates
    float       score;
    int         classId;
};

std::vector<Detection> postprocess(
    Ort::Value& outputTensor,
    int            origW,
    int            origH,
    int            modelW = 640,
    int            modelH = 640,
    float          confThreshold = 0.45f
);

struct OrtContext {
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    Ort::Session session;

    OrtContext(const wchar_t* modelPath)
        : env(ORT_LOGGING_LEVEL_WARNING, "YoloCropper"),
        session(nullptr)
    {
        sessionOptions.SetIntraOpNumThreads(4);
        sessionOptions.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session = Ort::Session(env, modelPath, sessionOptions);
    }
};


// Tests
void test_loadModel(int model);


// Process
std::vector<float> preprocess(const cv::Mat& letterboxed);


// Inference two results, the cropped image and the best detection
struct InferenceResult {
    std::optional<cv::Mat> croppedImage;
    std::optional<Detection> bestDetection;
};

InferenceResult runInference(const cv::Mat& original, const cv::Mat& letterboxed, OrtContext& ctx, bool DEBUG);


std::optional<cv::Mat> cropDetection(const cv::Mat& image, const std::vector<Detection>& detections);

