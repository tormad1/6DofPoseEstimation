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
    public Vector2 focalLengthPx = new Vector2(700f, 700f);
    public Vector2 principalPointPx = Vector2.zero;
    public Vector2 calibrationImageSizePx = new Vector2(640f, 480f);

    [Header("References")]
    public PoseManager poseManager;

    private bool runtimeReady = false;
    private bool loggedInvalidIntrinsics = false;
    private bool loggedFirstPose = false;
    private bool loggedEffectiveIntrinsics = false;
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
        if (!TryBuildCameraIntrinsics(fullFrameWidth, fullFrameHeight, out float[] cameraK))
            return;
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
            if (!loggedFirstPose)
            {
                Debug.Log(
                    "[PoseBridge] First ROI pose: " +
                    $"p=({pose.px:F4}, {pose.py:F4}, {pose.pz:F4}) " +
                    $"q=({pose.qx:F4}, {pose.qy:F4}, {pose.qz:F4}, {pose.qw:F4}) " +
                    $"confidence={pose.confidence:F3}"
                );
                loggedFirstPose = true;
            }
            if (poseManager != null)
                poseManager.OnPose(pose);
        }
        else if (result < 0)
        {
            Debug.LogWarning($"[PoseBridge] RunRoiPose failed: {outBuf}");
        }
    }


    private long GetTimestampUs()
    {
        return (stopwatch.ElapsedTicks * 1_000_000L) / System.Diagnostics.Stopwatch.Frequency;
    }

    private bool TryBuildCameraIntrinsics(
        int fullFrameWidth,
        int fullFrameHeight,
        out float[] cameraK)
    {
        cameraK = null;

        float fx = focalLengthPx.x;
        float fy = focalLengthPx.y;
        float cx = principalPointPx.x;
        float cy = principalPointPx.y;
        float calibWidth = calibrationImageSizePx.x;
        float calibHeight = calibrationImageSizePx.y;

        if (
            fx <= 0f || fy <= 0f || cx <= 0f || cy <= 0f ||
            calibWidth <= 0f || calibHeight <= 0f
        )
        {
            if (!loggedInvalidIntrinsics)
            {
                Debug.LogWarning(
                    "[PoseBridge] Manual webcam intrinsics are required. " +
                    $"Set focalLengthPx, principalPointPx and calibrationImageSizePx on PoseBridge. " +
                    $"Current values: fx={fx}, fy={fy}, cx={cx}, cy={cy}, " +
                    $"calibWidth={calibWidth}, calibHeight={calibHeight}. " +
                    $"Current full-frame size: {fullFrameWidth}x{fullFrameHeight}."
                );
                loggedInvalidIntrinsics = true;
            }
            return false;
        }

        float scaleX = fullFrameWidth / calibWidth;
        float scaleY = fullFrameHeight / calibHeight;
        fx *= scaleX;
        fy *= scaleY;
        cx *= scaleX;
        cy *= scaleY;

        loggedInvalidIntrinsics = false;
        cameraK = new[]
        {
            fx, 0f, cx,
            0f, fy, cy,
            0f, 0f, 1f
        };

        if (!loggedEffectiveIntrinsics)
        {
            Debug.Log(
                "[PoseBridge] Effective intrinsics: " +
                $"frame={fullFrameWidth}x{fullFrameHeight}, " +
                $"calibration={calibWidth}x{calibHeight}, " +
                $"fx={fx:F3}, fy={fy:F3}, cx={cx:F3}, cy={cy:F3}"
            );
            loggedEffectiveIntrinsics = true;
        }
        return true;
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
