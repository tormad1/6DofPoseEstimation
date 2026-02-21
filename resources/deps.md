# Dependencies required for dev

## ROI

*To install vcpkg*
- Package Manager for cpp.

```bash
cd C:\
git clone https://github.com/microsoft/vcpkg
cd vcpkg
bootstrap-vcpkg.bat
vcpkg integrate install
```
***
*OpenCV is allows for the use of YOLO models*
https://opencv.org/

```bash
cd C:\vcpkg
vcpkg install opencv4:x64-windows
```
***

*ONNXruntime was downloaded from the github releases page*
