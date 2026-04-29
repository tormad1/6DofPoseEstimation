using UnityEngine;

public class WebCam : MonoBehaviour
{
    [SerializeField] private Material webCamMaterial;
    public WebCamTexture Texture { get; private set; }

    void Start()
    {
        Application.runInBackground = true;
        Texture = new WebCamTexture();
        webCamMaterial.mainTexture = Texture;
        Texture.Play();
    }

    void OnDestroy() => Texture.Stop();
}