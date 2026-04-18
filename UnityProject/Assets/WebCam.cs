using UnityEngine;

public class WebCam : MonoBehaviour
{
    [SerializeField] private Material WebCamMaterial;
    //[SerializeField] private string webcamName = ""

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        WebCamDevice[] devices = WebCamTexture.devices;

        //Log out each webcam
        Debug.Log("Available webcams:");
        foreach(WebCamDevice webcam in devices)
        {
            Debug.Log(webcam.name);
        }

        //We get the webcam by name.
        //Note: on PC you can have multiple, but on mobile you can't
        //Can just call new() to get the deafult
        WebCamTexture webcamTexture = new();

        //Set our material texture to be our webcam texture.
        WebCamMaterial.mainTexture = webcamTexture;

        //Start the webcam:
        webcamTexture.Play();
          
    }
}
