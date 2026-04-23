#include "pch.h"
#include "ROIDll.h"
#include "crop.h"
#include "model.h"
#include "debug.h"


static OrtContext* g_ctx = nullptr;

extern "C" {
    int __cdecl ROI_Init(const wchar_t* modelPath)
    {
        if (g_ctx) return 1; // already initialised

        try {
            g_ctx = new OrtContext(modelPath);
            return 1;
        }
        catch (...) {
            return 0;
        }
    }


    int __cdecl ROI_ProcessFrame(
        const unsigned char* jpegData,
        int                  jpegLength,
        float* out_x,
        float* out_y,
        float* out_w,
        float* out_h,
        float* out_confidence)
    {
        // FAIL
        if (!g_ctx || !jpegData || jpegLength <= 0) return -1;

        // Decode JPEG bytes → cv::Mat
        cv::Mat image = loadImage_memory(jpegData, jpegLength);
        if (image.empty()) return -1;

        cv::Mat letterboxed = letterboxResizeImage(image, 640, 640);

        int result = runDetection(g_ctx, letterboxed, image, out_x, out_y, out_w, out_h, out_confidence);
        return result;
    }

    void __cdecl ROI_Shutdown()
    {
        delete g_ctx;
        g_ctx = nullptr;
    }

}