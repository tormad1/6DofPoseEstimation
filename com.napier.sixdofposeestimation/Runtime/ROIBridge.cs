using System;
using System.IO;
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
    public float lastX, lastY, lastW, lastH;
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
            out float outW, out float outH);

        [DllImport("ROIDLL", CallingConvention = CallingConvention.Cdecl)]
        public static extern int GetCroppedImage(out IntPtr outPtr, out int outSize);
    }

    void Start()
    {
        string modelsDir = Path.Combine(Application.streamingAssetsPath, "models");
        string[] candidateModelNames =
        {
            "yolov8m-canned.onnx",
            "yolov8m-can-large.onnx",
        };

        string modelPath = null;
        foreach (string candidateModelName in candidateModelNames)
        {
            string candidatePath = Path.Combine(modelsDir, candidateModelName);
            if (File.Exists(candidatePath))
            {
                modelPath = candidatePath;
                break;
            }
        }

        if (string.IsNullOrEmpty(modelPath))
        {
            Debug.LogError(
                $"[ROIBridge] No ROI detector model found in {modelsDir}. " +
                $"Tried: {string.Join(", ", candidateModelNames)}"
            );
            return;
        }

        int result = Native.InitROI(modelPath);
        Debug.Log($"[ROIBridge] InitROI: {result}");
        Debug.Log($"[ROIBridge] Model path: {modelPath}");
        Debug.Log($"[ROIBridge] File exists: {File.Exists(modelPath)}");
    }

    void OnDestroy() => Native.ShutdownROI();

    void Update()
    {
        frameCount++;
        if (frameCount < intervalFrames) return;
        frameCount = 0;

        WebCamTexture cam = webCam != null ? webCam.Texture : null;
        if (cam == null || !cam.isPlaying)
        {
            Debug.Log($"[ROIBridge] Skipping — webcam not ready. " +
                      $"Texture={cam != null}, " +
                      $"Playing={cam?.isPlaying}");
            return;
        }

        Debug.Log("[ROIBridge] Running detection...");

        ProcessFrame(cam);

        if (CroppedTexture != null && croppedMaterial != null)
            croppedMaterial.mainTexture = CroppedTexture;

        if (CroppedTexture != null && poseBridge != null)
        {
            poseBridge.SubmitRoi(
                CroppedTexture,
                lastX,
                lastY,
                lastW,
                lastH,
                cam.width,
                cam.height
            );
        }
    }

    private void ProcessFrame(WebCamTexture cam)
    {
        Color32[] pixels = cam.GetPixels32();
        byte[] raw = new byte[pixels.Length * 4];

        for (int i = 0; i < pixels.Length; i++)
        {
            raw[i * 4 + 0] = pixels[i].r;
            raw[i * 4 + 1] = pixels[i].g;
            raw[i * 4 + 2] = pixels[i].b;
            raw[i * 4 + 3] = pixels[i].a;
        }

        int status = Native.RunDetection(
            raw, cam.width, cam.height,
            out lastX, out lastY, out lastW, out lastH);

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
