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