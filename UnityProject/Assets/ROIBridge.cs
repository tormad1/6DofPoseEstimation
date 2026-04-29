using System;
using System.Runtime.InteropServices;
using UnityEngine;

public class ROIBridge : MonoBehaviour
{
    private static class Native
    {
        [DllImport("ROIDLL", CallingConvention = CallingConvention.Cdecl)]
        public static extern int InitROI(
            [MarshalAs(UnmanagedType.LPWStr)] string modelPath);

        [DllImport("ROIDLL", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ShutdownROI();

        [DllImport("ROIDLL", CallingConvention = CallingConvention.Cdecl)]
        public static extern int RunDetection(
            byte[] rgbaData, int width, int height,
            out float outX, out float outY,
            out float outW, out float outH,
            out float outScore);

        // outPtr points into the DLL's static buffer — copy immediately
        [DllImport("ROIDLL", CallingConvention = CallingConvention.Cdecl)]
        public static extern int GetCroppedImage(out IntPtr outPtr, out int outSize);
    }

    [Header("Detection")]
    public float lastX, lastY, lastW, lastH, lastScore;
    public Texture2D croppedTexture;
    public bool detectionFound = false;

    void Start()
    {
        string modelPath = System.IO.Path.Combine(
            Application.streamingAssetsPath, "models", "yolov8m-canned.onnx");

        int result = Native.InitROI(modelPath);
        Debug.Log($"[ROIBridge] InitROI: {result}");
    }

    void OnDestroy()
    {
        Native.ShutdownROI();
    }

    // Call this with the raw pixels from a WebCamTexture each frame
    public bool ProcessFrame(WebCamTexture cam)
    {
        // GetPixels32 returns RGBA, bottom-up — matches what ROIDLL expects
        Color32[] pixels = cam.GetPixels32();
        byte[] raw = new byte[pixels.Length * 4];
        Buffer.BlockCopy(pixels, 0, raw, 0, raw.Length);

        int status = Native.RunDetection(
            raw, cam.width, cam.height,
            out lastX, out lastY, out lastW, out lastH, out lastScore);

        detectionFound = (status == 1);

        if (detectionFound)
        {
            Debug.Log($"[ROIBridge] Box ({lastX:F0},{lastY:F0}) {lastW:F0}x{lastH:F0} conf={lastScore:F2}");
            FetchCroppedImage();
        }

        return detectionFound;
    }

    private void FetchCroppedImage()
    {
        if (Native.GetCroppedImage(out IntPtr ptr, out int size) != 1) return;

        // Copy out of DLL buffer immediately — ptr is invalid after next RunDetection
        byte[] jpegBytes = new byte[size];
        Marshal.Copy(ptr, jpegBytes, 0, size);

        if (croppedTexture == null)
            croppedTexture = new Texture2D(2, 2);

        croppedTexture.LoadImage(jpegBytes); // handles JPEG decode + resize
    }
}