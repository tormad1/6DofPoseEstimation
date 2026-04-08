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
