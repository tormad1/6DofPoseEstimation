#include "DummyPose.h"

#include <random>
#include <cmath>

static std::int64_t NowMicrosMonotonic()
{
    return NULL;
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
        return 0; //should not let exceptions cross the DLL boundary
    }
}