#include "DummyPose.h"

#include <cmath>
#include <limits>


extern "C" int __cdecl GetDummyPose(DummyPose* out_pose, std::int64_t timestamp_us, int mode)
{
    if (out_pose == nullptr) {
        return 0;
    }

    try
    {
        const double t = static_cast<double>(timestamp_us) * 1e-6;

        //base dummy confidence
        out_pose->confidence = 0.9f;

        //mode 1, low confidence frame
        if (mode == 1) {
            out_pose->confidence = 0.1f;
        }
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

        //mode 2, roatation spiked frame
        if (mode == 2) {
            const float jumped_half = half + 1.7f; //large step
            out_pose->qy = std::sin(jumped_half);
            out_pose->qw = std::cos(jumped_half);
        }

        //mode 3, position spiked frame
        if (mode == 3) {
            out_pose->px += 10.0f;
            out_pose->pz += 10.0f;
        }

        //mode 4, invalid rotation
        if (mode == 4) {
            out_pose->qx = 0.0f;
            out_pose->qy = 0.0f;
            out_pose->qz = 0.0f;
            out_pose->qw = 0.0f;
        }

        //mode 5, invalid numeric (NaN)
        if (mode == 5) {
            out_pose->px = std::numeric_limits<float>::quiet_NaN();
        }

        //mode 6, invalid numeric (Inf)
        if (mode == 6) {
            out_pose->px = std::numeric_limits<float>::infinity();
        }
        out_pose->timestamp_us = timestamp_us;
        return 1;
    }
    catch (...)
    {
        return 0; //should not let exceptions cross the dll boundary
    }
}
