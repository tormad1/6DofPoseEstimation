using UnityEngine;

public class WebCam : MonoBehaviour
{
    [SerializeField] private Material webCamMaterial;
    [SerializeField] private int requestedWidth = 640;
    [SerializeField] private int requestedHeight = 480;
    [SerializeField] private int requestedFps = 30;

    public WebCamTexture Texture { get; private set; }
    private bool loggedReady = false;

    void Start()
    {
        Application.runInBackground = true;
        Texture = new WebCamTexture(requestedWidth, requestedHeight, requestedFps);
        webCamMaterial.mainTexture = Texture;
        Texture.Play();
    }

    void Update()
    {
        if (loggedReady || Texture == null || !Texture.isPlaying)
            return;

        if (Texture.width > 16 && Texture.height > 16)
        {
            Debug.Log(
                $"[WebCam] Live resolution: {Texture.width}x{Texture.height} " +
                $"(requested {requestedWidth}x{requestedHeight} @ {requestedFps}fps)"
            );
            loggedReady = true;
        }
    }

    void OnDestroy() => Texture.Stop();
}
