#include "crop.hpp"
#include "model.hpp"
#include "test.hpp"
#include <iostream>
#include <filesystem>
#include <thread>
#include <chrono>

const std::string FRAME_PATH = "C:\\temp\\webcam_frame.jpg";

int main() {
    OrtContext ctx(L"resources/models/yolov8m-canned.onnx");

    std::filesystem::file_time_type lastProcessed;

    std::cout << "Waiting for frames at: " << FRAME_PATH << std::endl;

    while (true) {
        if (std::filesystem::exists(FRAME_PATH)) {
            auto modTime = std::filesystem::last_write_time(FRAME_PATH);

            if (modTime != lastProcessed) {
                lastProcessed = modTime;

                cv::Mat image = loadImage_abs(FRAME_PATH); // see note below
                if (!image.empty()) {
                    cv::Mat letterboxed = letterboxResizeImage(image, 640, 640);
                    cv::Mat cropped = runInference(image, letterboxed, ctx);

                    // Save with timestamp to avoid overwriting
                    auto ts = std::chrono::system_clock::now().time_since_epoch().count();
                    saveImage(cropped, "frame_" + std::to_string(ts) + ".jpg");
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    return 0;
}