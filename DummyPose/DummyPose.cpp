#include "DummyPose.h"

#include <random>
#include <cmath>
#include <chrono>
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
    try
    {
        return 1;
    }
    catch (...)
    {
        return 0; //should not let exceptions cross the dll boundary
    }
}