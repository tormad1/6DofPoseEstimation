using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;

public class PoseBridge : MonoBehaviour
{
    [Header("Mode")]
    public bool useDummy = false;
    public int dummyMode = 0;

    [Header("GigaPose")]
    public int cpuThreads = 4;
    public bool warmupRuntime = false;
    public int objectId = 1;
    public Texture2D croppedFrame;

    [Header("Camera Intrinsics")]
    public Camera sourceCamera;
    public bool useManualIntrinsics = false;
    public Vector2 focalLengthPx = new Vector2(700f, 700f);
    public Vector2 principalPointPx = Vector2.zero;
    public float fallbackVerticalFovDeg = 60f;

    [Header("References")]
    public PoseManager poseManager;

    private bool runtimeReady = false;
    private bool loggedFallbackIntrinsics = false;
    private System.Diagnostics.Stopwatch stopwatch;

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
        public static extern int GetDummyPose(ref Pose outPose, long timestampUs, int mode);
    }

    private static class GigaPoseBridgeNative
    {
        [DllImport("GigaPoseBridge", CallingConvention = CallingConvention.Cdecl)]
        public static extern int InitPython(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string pythonHome);

        [DllImport("GigaPoseBridge", CallingConvention = CallingConvention.Cdecl)]
        public static extern int InitGigaPoseRuntime(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string repoRoot,
            int cpuThreads,
            int warmup,
            StringBuilder outBuf,
            int outBufLen);

        [DllImport("GigaPoseBridge", CallingConvention = CallingConvention.Cdecl)]
        public static extern int RunRoiPose(
            byte[] rgbaData,
            int width,
            int height,
            int stride,
            float[] cameraK3x3,
            float bboxX,
            float bboxY,
            float bboxW,
            float bboxH,
            int objectId,
            long timestampUs,
            out Pose outPose,
            StringBuilder outBuf,
            int outBufLen);

        [DllImport("GigaPoseBridge", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ShutdownPython();
    }

    private static class WindowsNative
    {
        [DllImport("kernel32", SetLastError = true, CharSet = CharSet.Unicode)]
        public static extern bool SetDllDirectory(string path);
    }

    void Start()
    {
        stopwatch = System.Diagnostics.Stopwatch.StartNew();
        if (useDummy)
        {
            return;
        }

        string repoRoot = Path.GetFullPath(Path.Combine(Application.dataPath, "..", ".."));
        string pythonHome = Path.Combine(repoRoot, "gigaposeFork", ".python", "python-3.11.9-embed-amd64");
        string pluginDir = Path.Combine(repoRoot, "com.napier.sixdofposeestimation", "Plugins", "x86_64");
        string torchLib = Path.Combine(repoRoot, "gigaposeFork", ".venv", "Lib", "site-packages", "torch", "lib");

        string currentPath = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
        if (Directory.Exists(pluginDir) && !currentPath.Contains(pluginDir))
            currentPath = pluginDir + Path.PathSeparator + currentPath;
        if (Directory.Exists(torchLib) && !currentPath.Contains(torchLib))
            currentPath = torchLib + Path.PathSeparator + currentPath;
        Environment.SetEnvironmentVariable("PATH", currentPath);

        if (Directory.Exists(pluginDir))
            WindowsNative.SetDllDirectory(pluginDir);

        int initPythonResult = GigaPoseBridgeNative.InitPython(pythonHome);
        Debug.Log($"[PoseBridge] InitPython returned: {initPythonResult}");
        if (initPythonResult != 1)
        {
            return;
        }

        var initMessage = new StringBuilder(512);
        int initRuntimeResult = GigaPoseBridgeNative.InitGigaPoseRuntime(
            repoRoot,
            cpuThreads,
            warmupRuntime ? 1 : 0,
            initMessage,
            initMessage.Capacity
        );
        runtimeReady = initRuntimeResult == 1;
        Debug.Log($"[PoseBridge] InitGigaPoseRuntime returned: {initRuntimeResult} ({initMessage})");
    }

    void OnDestroy()
    {
        if (!useDummy)
        {
            GigaPoseBridgeNative.ShutdownPython();
            Debug.Log("[PoseBridge] Python shut down.");
        }
    }

    void Update()
    {
        if (!useDummy)
        {
            return;
        }

        int status = GetPose(out var pose);
        if (status == 1 && poseManager != null)
            poseManager.OnPose(pose);
    }

    public int GetPose(out Pose pose)
    {
        long timestampUs = GetTimestampUs();
        pose = default;
        if (useDummy)
            return DummyPoseNative.GetDummyPose(ref pose, timestampUs, dummyMode);
        return 0;
    }

    public void SubmitRoi(
        Texture2D roiTexture,
        float bboxX,
        float bboxY,
        float bboxW,
        float bboxH,
        int fullFrameWidth,
        int fullFrameHeight)
    {
        croppedFrame = roiTexture;

        if (useDummy || !runtimeReady || roiTexture == null)
            return;
        if (fullFrameWidth <= 0 || fullFrameHeight <= 0)
        {
            Debug.LogWarning("[PoseBridge] Invalid full-frame size for ROI submission.");
            return;
        }

        byte[] rgbaBytes = TextureToRgbaBytes(roiTexture);
        float[] cameraK = BuildCameraIntrinsics(fullFrameWidth, fullFrameHeight);
        long timestampUs = GetTimestampUs();

        var outBuf = new StringBuilder(512);
        int result = GigaPoseBridgeNative.RunRoiPose(
            rgbaBytes,
            roiTexture.width,
            roiTexture.height,
            roiTexture.width * 4,
            cameraK,
            bboxX,
            bboxY,
            bboxW,
            bboxH,
            objectId,
            timestampUs,
            out Pose pose,
            outBuf,
            outBuf.Capacity
        );

        if (result == 1)
        {
            if (poseManager != null)
                poseManager.OnPose(pose);
        }
        else if (result < 0)
        {
            Debug.LogWarning($"[PoseBridge] RunRoiPose failed: {outBuf}");
        }
    }

    public void SendFrameToPython(Texture2D frame)
    {
        if (frame == null)
            return;

        SubmitRoi(
            frame,
            0f,
            0f,
            frame.width,
            frame.height,
            frame.width,
            frame.height
        );
    }

    private long GetTimestampUs()
    {
        return (stopwatch.ElapsedTicks * 1_000_000L) / System.Diagnostics.Stopwatch.Frequency;
    }

    private float[] BuildCameraIntrinsics(int fullFrameWidth, int fullFrameHeight)
    {
        float fx;
        float fy;
        float cx;
        float cy;

        if (useManualIntrinsics)
        {
            fx = focalLengthPx.x;
            fy = focalLengthPx.y > 0f ? focalLengthPx.y : fx;
            cx = principalPointPx.x > 0f ? principalPointPx.x : fullFrameWidth * 0.5f;
            cy = principalPointPx.y > 0f ? principalPointPx.y : fullFrameHeight * 0.5f;
        }
        else
        {
            float verticalFovDeg = sourceCamera != null ? sourceCamera.fieldOfView : fallbackVerticalFovDeg;
            if (sourceCamera == null && !loggedFallbackIntrinsics)
            {
                Debug.LogWarning("[PoseBridge] sourceCamera not set. Using fallback FOV for intrinsics.");
                loggedFallbackIntrinsics = true;
            }

            float verticalFovRad = verticalFovDeg * Mathf.Deg2Rad;
            fy = 0.5f * fullFrameHeight / Mathf.Tan(0.5f * verticalFovRad);
            fx = fy;
            cx = fullFrameWidth * 0.5f;
            cy = fullFrameHeight * 0.5f;
        }

        return new[]
        {
            fx, 0f, cx,
            0f, fy, cy,
            0f, 0f, 1f
        };
    }

    private static byte[] TextureToRgbaBytes(Texture2D texture)
    {
        Color32[] pixels = texture.GetPixels32();
        byte[] rgbaBytes = new byte[pixels.Length * 4];

        for (int i = 0; i < pixels.Length; i++)
        {
            int offset = i * 4;
            rgbaBytes[offset + 0] = pixels[i].r;
            rgbaBytes[offset + 1] = pixels[i].g;
            rgbaBytes[offset + 2] = pixels[i].b;
            rgbaBytes[offset + 3] = pixels[i].a;
        }

        return rgbaBytes;
    }
}
