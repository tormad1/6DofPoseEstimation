#include "pch.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <optional>

#include "crop.h"
#include "model.h"
#include "debug.h"


void test_loadModel(int model) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YoloCropper");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(4);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    if (model == 0){
        sessionOptions.SetOptimizedModelFilePath(L"resources/models/yolov8m_optimized.onnx");
        Ort::Session session(env, L"resources/models/yolov8m.onnx", sessionOptions);
        std::cout << "Model Loaded" << std::endl;
	}

    else if (model == 1) {
        sessionOptions.SetOptimizedModelFilePath(L"resources/models/yolo8m-canned.onnx");
        Ort::Session session(env, L"resources/models/yolov8m-canned.onnx", sessionOptions);
        std::cout << "Model Loaded" << std::endl;
    }

    else{
        std::cout << "Invalid model index" << std::endl;
    }
}


std::vector<float> preprocess(const cv::Mat& letterboxed) {
    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);  // BGR -> RGB

    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);           // Normalize to [0,1]

    // HWC -> CHW (what ONNX/YOLO expects)
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    std::vector<float> tensor;
    tensor.reserve(3 * 640 * 640);
    for (auto& ch : channels)
        tensor.insert(tensor.end(), (float*)ch.datastart, (float*)ch.dataend);

    return tensor;
}


struct InferenceResult runInference(const cv::Mat& original, const cv::Mat& letterboxed, OrtContext& ctx, bool DEBUG) {


    std::vector<float> inputTensor = preprocess(letterboxed);

    // Shape is [image, colour channels, height, width]
    // [1, 3, 640, 640]
    std::vector<int64_t> inputShape = { 1, 3, letterboxed.rows, letterboxed.cols };

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );

    // Wrap your raw float vector in an OrtValue tensor
    Ort::Value inputOrtTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputTensor.data(),       
        inputTensor.size(),       
        inputShape.data(),        
        inputShape.size()         
    );

    // Get input/output node names from the model
    Ort::AllocatorWithDefaultOptions allocator;

    auto inputNamePtr = ctx.session.GetInputNameAllocated(0, allocator);
    auto outputNamePtr = ctx.session.GetOutputNameAllocated(0, allocator);

    const char* inputName = inputNamePtr.get();
    const char* outputName = outputNamePtr.get();

    print_DEBUG("Input name:  " + *inputName, DEBUG);
    print_DEBUG("Output name: " + *outputName, DEBUG);

    // Run
    auto outputTensors = ctx.session.Run(
        Ort::RunOptions{ nullptr },
        &inputName,        
        &inputOrtTensor,   
        1,                 
        &outputName,       
        1                  
    );


    // Postprocess and crop
    std::vector<Detection> detections = postprocess(
        outputTensors[0],
        original.cols,
        original.rows
    );



    // Save cropped image and coords of detection.
    auto cropped = cropDetection(original, detections);
    InferenceResult result;
    result.croppedImage = cropped;
    if (!detections.empty()) {
        result.bestDetection = detections[0];
    }
    return result;
}

std::vector<Detection> postprocess(Ort::Value& outputTensor, int origW, int origH, int modelW, int modelH, float confThreshold){
    // Get raw pointer to the output data
    float* data = outputTensor.GetTensorMutableData<float>();
    auto typeInfo = outputTensor.GetTensorTypeAndShapeInfo();
    auto outputShape = typeInfo.GetShape();


    const int64_t numBoxes = outputShape[2];
    const int64_t numClasses = outputShape[1];

    // For custom model, the class is 0 to detect cans.
	// For the original model, the class is 39 to detect bottles.
    const int targetClass = 0;

    // Work out the letterbox scale
    float scale = std::min(
        static_cast<float>(modelW) / origW,
        static_cast<float>(modelH) / origH
    );
    float padX = (modelW - origW * scale) / 2.0f;
    float padY = (modelH - origH * scale) / 2.0f;

    // Collect candidate boxes before NMS
    std::vector<cv::Rect>  boxes;
    std::vector<float>     scores;
    std::vector<int>       classIds;

    for (int i = 0; i < numBoxes; i++) {

        // Find the highest scoring class for this box
        float bestScore = 0.0f;
        int   bestClass = -1;
        for (int c = 0; c < numClasses; c++) {
            float score = data[(4 + c) * numBoxes + i];
            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }

        // Skip if below threshold or not a can
        if (bestScore < confThreshold) continue;
        if (bestClass != targetClass)  continue;

        // Decode box from center format (cx,cy,w,h) in 640x640 space
        float cx = data[0 * numBoxes + i];
        float cy = data[1 * numBoxes + i];
        float w = data[2 * numBoxes + i];
        float h = data[3 * numBoxes + i];


        // Convert from letterboxed 640x640 back to original image coords
        float x1 = (cx - w / 2.0f - padX) / scale;
        float y1 = (cy - h / 2.0f - padY) / scale;
        float x2 = (cx + w / 2.0f - padX) / scale;
        float y2 = (cy + h / 2.0f - padY) / scale;

        // Clamp to image bounds
        x1 = std::max(0.0f, std::min(x1, (float)origW));
        y1 = std::max(0.0f, std::min(y1, (float)origH));
        x2 = std::max(0.0f, std::min(x2, (float)origW));
        y2 = std::max(0.0f, std::min(y2, (float)origH));

        boxes.push_back(cv::Rect(
            cv::Point((int)x1, (int)y1),
            cv::Point((int)x2, (int)y2)
        ));
        scores.push_back(bestScore);
        classIds.push_back(bestClass);
    }

    std::vector<int> kept;
    cv::dnn::NMSBoxes(boxes, scores, confThreshold, 0.45f, kept);

    std::vector<Detection> detections;
    for (int idx : kept) {
        Detection d;
        d.box = boxes[idx];
        d.score = scores[idx];
        d.classId = classIds[idx];
        detections.push_back(d);
    }



    std::cout << "Detections after NMS: " << detections.size() << std::endl;
    for (auto& d : detections) {
        std::cout << "  class=" << d.classId
            << " score=" << d.score
            << " box=" << d.box << std::endl;
    }

    return detections;
}

std::optional<cv::Mat> cropDetection(const cv::Mat& image, const std::vector<Detection>& detections) {
    if (detections.empty()) {
        //std::cout << "No detections, returning original image" << std::endl;
        return std::nullopt;
    }

    // Pick highest confidence box
    const Detection* best = &detections[0];
    for (auto& d : detections)
        if (d.score > best->score) best = &d;

    return image(best->box).clone();
}

