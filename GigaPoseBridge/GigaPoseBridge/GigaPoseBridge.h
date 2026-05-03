#pragma once
#include <cstdint>

#ifdef _WIN32
#define GIGAPOSE_EXPORT __declspec(dllexport)
#else
#define GIGAPOSE_EXPORT
#endif

extern "C" {

    struct GigaPoseNativePose
    {
        float px;
        float py;
        float pz;
        float qx;
        float qy;
        float qz;
        float qw;
        float confidence;
        int64_t timestamp_us;
    };

    // Initialises the embedded Python interpreter
    // python_home: path to the embedded CPython home directory
    // Returns 1 on success, 0 on failure
    GIGAPOSE_EXPORT int __cdecl InitPython(const char* python_home);

    // Shuts down the embedded Python interpreter
    // Call this when Unity unloads the plugin
    GIGAPOSE_EXPORT void __cdecl ShutdownPython();

    // passes image_path to Python, which opens it with Pillow
    // and writes the result into out_buf as a null-terminated string
    // out_buf_len: size of the buffer you're passing in
    // Returns 1 on success, 0 on failure
    GIGAPOSE_EXPORT int __cdecl OpenImageTest(const char* image_path, char* out_buf, int out_buf_len);

    // Creates the CPU runtime once and keeps it alive inside the embedded interpreter.
    // repo_root should be the parent folder that contains gigaposeFork.
    GIGAPOSE_EXPORT int __cdecl InitGigaPoseRuntime(
        const char* repo_root,
        int cpu_threads,
        int warmup,
        char* out_buf,
        int out_buf_len
    );

    // Runs top-1 ROI inference on one RGBA crop.
    // Returns 1 when a pose is produced, 0 when no pose is produced, -1 on error.
    GIGAPOSE_EXPORT int __cdecl RunRoiPose(
        const uint8_t* rgba_data,
        int width,
        int height,
        int stride,
        const float* camera_k_3x3,
        float bbox_x,
        float bbox_y,
        float bbox_w,
        float bbox_h,
        int object_id,
        int64_t timestamp_us,
        GigaPoseNativePose* out_pose,
        char* out_buf,
        int out_buf_len
    );

}
