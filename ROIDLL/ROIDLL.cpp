// ROIDLL.cpp
#include "pch.h"
#include "model.h"
#include "crop.h"

static OrtContext* g_ctx = nullptr;
static std::vector<uint8_t> g_croppedBuffer; // JPEG bytes, kept in memory

extern "C" {

    __declspec(dllexport) int __cdecl InitROI(const wchar_t* modelPath)
    {
        try {
            g_ctx = new OrtContext(modelPath);
            return 1;
        }
        catch (...) { return 0; }
    }

    __declspec(dllexport) void __cdecl ShutdownROI()
    {
        delete g_ctx;
        g_ctx = nullptr;
        g_croppedBuffer.clear();
    }

    // Accepts raw RGBA pixels from Unity (GetPixels32 gives RGBA)
    // Returns 1 if detection found, 0 if not
    __declspec(dllexport) int __cdecl RunDetection(
        const uint8_t* rgbaData, int width, int height,
        float* outX, float* outY, float* outW, float* outH)
    {
        if (!g_ctx) return -1;

        // RGBA -> BGR cv::Mat
        cv::Mat rgba(height, width, CV_8UC4, (void*)rgbaData);
        cv::Mat bgr;
        cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
        cv::flip(bgr, bgr, 0); // Unity textures are flipped vertically

        cv::Mat letterboxed = letterboxResizeImage(bgr, 640, 640);
        auto inferenceResult = runInference(bgr, letterboxed, *g_ctx, false);

        if (!inferenceResult.croppedImage.has_value()) return 0;

        // Store cropped image as JPEG in static buffer
        std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 90 };
        cv::imencode(".jpg", inferenceResult.croppedImage.value(), g_croppedBuffer, params);

        // bounding box
        *outX = inferenceResult.bestDetection->box.x;
        *outY = inferenceResult.bestDetection->box.y;
        *outW = inferenceResult.bestDetection->box.width;
        *outH = inferenceResult.bestDetection->box.height;
        return 1;
    }

    
    __declspec(dllexport) int __cdecl GetCroppedImage(const uint8_t** outPtr, int* outSize)
    {
        if (g_croppedBuffer.empty()) return 0;
        *outPtr = g_croppedBuffer.data();
        *outSize = (int)g_croppedBuffer.size();
        return 1;
    }

} // extern "C"