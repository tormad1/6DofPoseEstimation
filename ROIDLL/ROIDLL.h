#pragma once
extern "C" {
    // Call once at startup — loads the ONNX model
    __declspec(dllexport) int __cdecl ROI_Init(const wchar_t* modelPath);

    // Pass raw JPEG bytes from C#, get back the box coords
    // Returns 1 on detection, 0 on no detection, -1 on error
    __declspec(dllexport) int __cdecl ROI_ProcessFrame(
        const unsigned char* jpegData,
        int                  jpegLength,
        float* out_x,
        float* out_y,
        float* out_w,
        float* out_h,
        float* out_confidence
    );

    __declspec(dllexport) void __cdecl ROI_Shutdown();
}