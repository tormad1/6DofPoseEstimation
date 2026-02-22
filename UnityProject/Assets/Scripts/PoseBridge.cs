using System.Diagnostics;
using System.Runtime.InteropServices;
using UnityEngine;

public class PoseBridge : MonoBehaviour
{
    public bool useDummy = true;
    public Texture2D croppedFrame;
    public int mode = 0;
    public int intervalFrames = 10;
    private int frameCount = 0;
    private Stopwatch stopwatch;
    public PoseManager poseManager;

    [StructLayout(LayoutKind.Sequential)]
    public struct Pose
    {
        public float px, py, pz;
        public float qx, qy, qz, qw;
        public float confidence;
        public long timestamp_us;
    }

    private static class DummyPoseNative
    {
        [DllImport("DummyPose", CallingConvention = CallingConvention.Cdecl)]
        static public extern int GetDummyPose(ref Pose out_pose, long timestamp_us, int mode);
    }

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        stopwatch = Stopwatch.StartNew();
    }
    void Update()
    {
        frameCount++;
        if (useDummy && frameCount == intervalFrames)
        {
            frameCount=0;
            int status = GetPose(out var pose);
            //Debug.Log(status);
            if (status == 1 && poseManager != null)
            {
                poseManager.OnPose(pose);
            }
        }
    }
    // Update is called once per frame
    public int GetPose(out Pose pose)
    {
        long timestampMicro = (stopwatch.ElapsedTicks * 1_000_000L) / Stopwatch.Frequency;
        pose = default;

        if (useDummy)
        {
            return DummyPoseNative.GetDummyPose(ref pose, timestampMicro, mode);
        }

        //return RealPoseNative.GetDummyPose(ref pose, timestampMicro, mode);
        //NOT IMPLEMENTED YET

        return 0;
    }
}
