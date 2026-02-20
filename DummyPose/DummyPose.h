#pragma once

#include <cstdint>

#ifdef _WIN32
#define DUMMYPOSE_EXPORT __declspec(dllexport)
#else
#define DUMMYPOSE_EXPORT
#endif

extern "C" {

    //data
    struct DummyPose
    {
        float px, py, pz; //pos
        float qx, qy, qz, qw; //rotation quaternion
        float confidence; //from 0 to 1
        std::int64_t timestamp_us;
    };

    //will return 1 on success
    DUMMYPOSE_EXPORT int __cdecl GetDummyPose(DummyPose* out_pose);
    DUMMYPOSE_EXPORT int __cdecl SetDummyPoseMode(int mode);
    DUMMYPOSE_EXPORT int __cdecl ResetDummyPose();
}