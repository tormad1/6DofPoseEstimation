#include "crop.hpp"
#include "model.hpp"
#include "test.hpp"
#include <iostream>
#include <filesystem>
#include <thread>
#include <chrono>

const std::string FRAME_PATH = "C:\\temp\\webcam_frame.jpg";

int main() {
    // Preload onnx env.
    OrtContext ctx(L"resources/models/yolov8m-canned.onnx");

    std::filesystem::file_time_type lastProcessed;
    std::cout << "Waiting for frames at: " << FRAME_PATH << std::endl;

    while (true) {
        if (std::filesystem::exists(FRAME_PATH)) {
            auto modTime = std::filesystem::last_write_time(FRAME_PATH);

            if (modTime != lastProcessed) {
                lastProcessed = modTime;

                cv::Mat image = loadImage_abs(FRAME_PATH);
                if (!image.empty()) {
                    cv::Mat letterboxed = letterboxResizeImage(image, 640, 640);
                    cv::Mat cropped = runInference(image, letterboxed, ctx);

                    // Save with timestamp to avoid overwriting
					saveImage_ts(cropped, "cropped.jpg");
                }
                else {
					std::cout << "Failed to load image from: " << FRAME_PATH << std::endl;
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}