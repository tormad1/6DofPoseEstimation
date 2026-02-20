#include "DummyPose.h"

#include <random>
#include <cmath>
#include <chrono>

static std::int64_t g_t0_us = 0;
static std::int64_t g_last_ts_us = 0;
static int g_mode = 0;
static std::int64_t g_i = 0;

static std::int64_t NowMicrosMonotonic()
{
    using clock = std::chrono::steady_clock;
    auto now = clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
}

static void RandomUnitQuaternion()
{

}

extern "C" int __cdecl GetDummyPose(DummyPose* out_pose)
{
    if (out_pose == nullptr) {
        return 0;
    }

    if (g_t0_us == 0) {
        g_t0_us = NowMicrosMonotonic();
        g_last_ts_us = g_t0_us - 1;
        g_i = 0;
        g_mode = 0;
    }

    try
    {
        const std::int64_t now_us = NowMicrosMonotonic();
        const double t = (now_us - g_t0_us) * 1e-6; //since start in seconds

        //dummy change for postion 
        out_pose->px = 0.5f * static_cast<float>(std::sin(t));
        out_pose->py = 0.1f * static_cast<float>(std::sin(t * 0.2)); //t * 0.2 changes the speed while the 0.2f * sets the amplitude
        out_pose->pz = 0.3f * static_cast<float>(std::sin(t * 1.3));

        //dummy change for rotation
        const float half = static_cast<float>(t * 0.4);
        out_pose->qx = 0.0f;
        out_pose->qy = std::sin(half);
        out_pose->qz = 0.0f;
        out_pose->qw = std::cos(half);

        //base dummy confidence
        out_pose->confidence = 0.9f;

        //timestamp increasing
        std::int64_t ts = now_us;
        if (ts <= g_last_ts_us) {
            ts = g_last_ts_us + 1;
        }
        out_pose->timestamp_us = ts;
        g_last_ts_us = ts;

        ++g_i;
        return 1;
    }
    catch (...)
    {
        return 0; //should not let exceptions cross the dll boundary
    }
}

extern "C" int __cdecl SetDummyPoseMode(int mode)
{
    try {
        g_mode = mode;
        return 1;
    }
    catch (...) {
        return 0;
    }
}

extern "C" int __cdecl ResetDummyPose()
{
    try {
        g_t0_us = 0; //makes init in GetDummyPose happen again
        g_last_ts_us = 0;
        g_i = 0;
        g_mode = 0;
        return 1;
    }
    catch (...) {
        return 0;
    }
}