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

    GIGAPOSE_EXPORT int __cdecl InitPython(const char* python_home);
    GIGAPOSE_EXPORT void __cdecl ShutdownPython();
    GIGAPOSE_EXPORT int __cdecl OpenImageTest(const char* image_path, char* out_buf, int out_buf_len);
    GIGAPOSE_EXPORT int __cdecl InitGigaPoseRuntime(
        const char* repo_root,
        int cpu_threads,
        int warmup,
        char* out_buf,
        int out_buf_len
    );
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
