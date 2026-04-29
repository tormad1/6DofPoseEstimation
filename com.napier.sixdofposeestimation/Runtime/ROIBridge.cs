using System;
using System.Runtime.InteropServices;
using UnityEngine;

public class ROIBridge : MonoBehaviour
{
    [Header("References")]
    public WebCam webCam;
    public PoseBridge poseBridge;

    [Header("Settings")]
    public int intervalFrames = 10;

    [Header("Debug")]
    public float lastX, lastY, lastW, lastH, lastScore;
    public bool detectionFound = false;

    private int frameCount = 0;

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

        [DllImport("ROIDLL", CallingConvention = CallingConvention.Cdecl)]
        public static extern int GetCroppedImage(out IntPtr outPtr, out int outSize);
    }

    void Start()
    {
        string modelPath = System.IO.Path.Combine(
            Application.streamingAssetsPath, "models", "yolov8m-canned.onnx");

        int result = Native.InitROI(modelPath);
        Debug.Log($"[ROIBridge] InitROI: {result}");
    }

    void OnDestroy() => Native.ShutdownROI();

    void Update()
    {
        frameCount++;
        if (frameCount < intervalFrames) return;
        frameCount = 0;

        if (webCam == null || webCam.Texture == null || !webCam.Texture.didUpdateThisFrame)
            return;

        Texture2D cropped = ProcessFrame(webCam.Texture);

        if (cropped != null && poseBridge != null)
            poseBridge.croppedFrame = cropped;
    }

    private Texture2D ProcessFrame(WebCamTexture cam)
    {
        Color32[] pixels = cam.GetPixels32();
        byte[] raw = new byte[pixels.Length * 4];
        Buffer.BlockCopy(pixels, 0, raw, 0, raw.Length);

        int status = Native.RunDetection(
            raw, cam.width, cam.height,
            out lastX, out lastY, out lastW, out lastH, out lastScore);

        detectionFound = (status == 1);
        if (!detectionFound) return null;

        Debug.Log($"[ROIBridge] Box ({lastX:F0},{lastY:F0}) {lastW:F0}x{lastH:F0} conf={lastScore:F2}");

        if (Native.GetCroppedImage(out IntPtr ptr, out int size) != 1) return null;

        byte[] jpegBytes = new byte[size];
        Marshal.Copy(ptr, jpegBytes, 0, size);

        Texture2D tex = new Texture2D(2, 2);
        tex.LoadImage(jpegBytes);
        return tex;
    }
}