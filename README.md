# 6DofPoseEstimation

## ROI
The ROI will be handled by using a YOLO model, through the onnxruntime library.
The model will first be downloaded, but the goal is to have a trained model on our own dataset, which will be trained using the Ultralytics YOLOv8 framework.

### Notes
Input image must be transformed into a 640x640 image.
roi cropping is layed out in a format of (x, y, w, h).


After passing through YOLO, you will be given crop coords.
Reverse the cropping and scale the coords back to the original image
for max resolution.


### References
https://universe.roboflow.com/project-qynru/can-gcc8l
This is the model i will be using. The online inferencing detected the images properly.
## Pose Bridge
Pose bridge takes the cropped ROI image produced by the ROI stage and passes it into the pose estimation DLL to generate a pose for the frame given. the resulting pose is then passed into Unity through the PoseManager script.
### Notes 
Pose Bridge is currently not completed, only a dummy mode exist for testing without the real pose dll

## Dummy Pose DLL
Dummy pose is a native dll used for testing purposes while the real pose dll is not ready. 
it takes a timestamp and writes a fake pose into the output struct, with position and rotation chnaging over time.
### Notes 
the interger mode is used to force specifc cases: mode 1 outputs a low confidence frame, mode 2 spikes rotation, mode 3 spikes position, mode 4 outputs an invalid rotation, mode 5 forces NaN, mode 6 forces infinity.
