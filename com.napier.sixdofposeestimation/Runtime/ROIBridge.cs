using System;
using System.Runtime.InteropServices;
using UnityEngine;


public class ROIBridge : MonoBehaviour
{
    [Header("References")]
    public WebCam webCam;
    public PoseBridge poseBridge;
    [SerializeField] private Material croppedMaterial;

    [Header("Settings")]
    public int intervalFrames = 10;

    [Header("Debug")]
    public float lastX, lastY, lastW, lastH, lastScore;
    public bool detectionFound = false;

    private int frameCount = 0;
    public Texture2D CroppedTexture { get; private set; }

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

        Debug.Log($"[ROIBridge] Model path: {modelPath}");
        Debug.Log($"[ROIBridge] File exists: {System.IO.File.Exists(modelPath)}");
    }

    void OnDestroy() => Native.ShutdownROI();

    void Update()
    {
        frameCount++;
        if (frameCount < intervalFrames) return;
        frameCount = 0;

        if (webCam == null || webCam.Texture == null || !webCam.Texture.isPlaying)
        {
            Debug.Log($"[ROIBridge] Skipping — webcam not ready. " +
                      $"Texture={webCam?.Texture != null}, " +
                      $"Playing={webCam?.Texture?.isPlaying}");
            return;
        }

        Debug.Log("[ROIBridge] Running detection...");

        ProcessFrame(webCam.Texture);

        // Update material directly using this class's own CroppedTexture
        if (CroppedTexture != null && croppedMaterial != null)
            croppedMaterial.mainTexture = CroppedTexture;

        // Hand texture to PoseBridge for GigaPose
        if (CroppedTexture != null && poseBridge != null)
            poseBridge.croppedFrame = CroppedTexture;
    }

    private void ProcessFrame(WebCamTexture cam)
    {
        Color32[] pixels = cam.GetPixels32();
        byte[] raw = new byte[pixels.Length * 4];

        // Replace Buffer.BlockCopy with manual extraction
        for (int i = 0; i < pixels.Length; i++)
        {
            raw[i * 4 + 0] = pixels[i].r;
            raw[i * 4 + 1] = pixels[i].g;
            raw[i * 4 + 2] = pixels[i].b;
            raw[i * 4 + 3] = pixels[i].a;
        }

        int status = Native.RunDetection(
            raw, cam.width, cam.height,
            out lastX, out lastY, out lastW, out lastH, out lastScore);

        detectionFound = (status == 1);
        if (!detectionFound) return;

        if (Native.GetCroppedImage(out IntPtr ptr, out int size) != 1) return;

        byte[] jpegBytes = new byte[size];
        Marshal.Copy(ptr, jpegBytes, 0, size);

        if (CroppedTexture == null)
            CroppedTexture = new Texture2D(2, 2);

        CroppedTexture.LoadImage(jpegBytes);
    }
}