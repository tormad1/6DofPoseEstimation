using System.IO;
using System.Runtime.InteropServices;
using UnityEngine;

public class PoseBridge : MonoBehaviour
{
    public bool useDummy = false;
    public Texture2D croppedFrame;
    private int mode = 0;
    public int intervalFrames = 10;
    private int frameCount = 0;
    private System.Diagnostics.Stopwatch stopwatch;
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
        public static extern int GetDummyPose(ref Pose out_pose, long timestamp_us, int mode);
    }

    private static class GigaPoseBridgeNative
    {
        [DllImport("GigaPoseBridge", CallingConvention = CallingConvention.Cdecl)]
        public static extern int InitPython(string python_home);

        [DllImport("GigaPoseBridge", CallingConvention = CallingConvention.Cdecl)]
        public static extern int OpenImageTest(string image_path, System.Text.StringBuilder out_buf, int out_buf_len);

        [DllImport("GigaPoseBridge", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ShutdownPython();
    }

    void Start()
    {
        stopwatch = System.Diagnostics.Stopwatch.StartNew();

        string pythonHome = @"C:\Users\solar\miniconda3";
        int initResult = GigaPoseBridgeNative.InitPython(pythonHome);
        Debug.Log($"[PoseBridge] InitPython returned: {initResult}");
    }

    void OnDestroy()
    {
        GigaPoseBridgeNative.ShutdownPython();
        Debug.Log("[PoseBridge] Python shut down.");
    }

    void Update()
    {
        frameCount++;
        if (frameCount == intervalFrames)
        {
            frameCount = 0;

            if (useDummy)
            {
                int status = GetPose(out var pose);
                if (status == 1 && poseManager != null)
                    poseManager.OnPose(pose);
            }
            else if (croppedFrame != null)
            {
                SendFrameToPython(croppedFrame);
            }
        }

      
    }

    public int GetPose(out Pose pose)
    {
        long timestampMicro = (stopwatch.ElapsedTicks * 1_000_000L) / System.Diagnostics.Stopwatch.Frequency;
        pose = default;
        if (useDummy)
            return DummyPoseNative.GetDummyPose(ref pose, timestampMicro, mode);
        return 0;
    }

    public void SendFrameToPython(Texture2D frame)
    {
        byte[] pngBytes = frame.EncodeToPNG();

        // Write to a temp file so OpenImageTest (file-path API) can read it
        string tempPath = Path.Combine(Application.temporaryCachePath, "frame.png");
        File.WriteAllBytes(tempPath, pngBytes);

        var outBuf = new System.Text.StringBuilder(256);
        int result = GigaPoseBridgeNative.OpenImageTest(tempPath, outBuf, 256);
        Debug.Log($"[PoseBridge] Python opened frame: {outBuf} (status={result})");
    }
}