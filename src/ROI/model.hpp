#pragma once
#include <onnxruntime_cxx_api.h>


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

cv::Mat cropDetection(const cv::Mat& image, const std::vector<Detection>& detections);

cv::Mat cropDetection(const cv::Mat& image, const std::vector<Detection>& detections);
void test_loadModel(int model);
std::vector<float> preprocess(const cv::Mat& letterboxed);
cv::Mat runInference(const cv::Mat& original, const cv::Mat& letterboxed);
std::vector<Detection> postprocess(Ort::Value& outputTensor, int origW, int origH, int modelW, int modelH, float confThreshold);
cv::Mat cropDetection(const cv::Mat& image, const std::vector<Detection>& detections);