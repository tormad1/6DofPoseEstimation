using UnityEngine;
using System.IO;
using System.Collections;

public class WebCam : MonoBehaviour
{
    [SerializeField] private Material WebCamMaterial;
    [SerializeField] private float captureIntervalSeconds = 2.0f;

    // Must match what your C++ code polls
    private string outputPath = @"C:\temp\webcam_frame.jpg";
    private string outputPathTmp = @"C:\temp\webcam_frame.tmp.jpg";

    private WebCamTexture webcamTexture;
    private Texture2D captureBuffer;

    void Start()
    {
        Application.runInBackground = true;
        webcamTexture = new WebCamTexture();
        WebCamMaterial.mainTexture = webcamTexture;
        webcamTexture.Play();

        // Ensure output dir exists
        Directory.CreateDirectory(@"C:\temp");

        StartCoroutine(CaptureLoop());
    }

    IEnumerator CaptureLoop()
    {
        // Wait for webcam to actually start
        while (!webcamTexture.didUpdateThisFrame)
            yield return null;

        while (true)
        {
            CaptureFrame();
            yield return new WaitForSeconds(captureIntervalSeconds);
        }
    }

    void CaptureFrame()
    {
        try
        {
            if (captureBuffer == null || captureBuffer.width != webcamTexture.width)
                captureBuffer = new Texture2D(webcamTexture.width, webcamTexture.height);

            captureBuffer.SetPixels(webcamTexture.GetPixels());
            captureBuffer.Apply();

            byte[] jpg = ImageConversion.EncodeToJPG(captureBuffer, 90);

            File.WriteAllBytes(outputPathTmp, jpg);
            File.Copy(outputPathTmp, outputPath, overwrite: true);

            Debug.Log("Frame captured at " + System.DateTime.Now.ToString("HH:mm:ss"));
        }
        catch (System.Exception e)
        {
            Debug.LogError("CaptureFrame failed: " + e.Message);
        }
    }
}