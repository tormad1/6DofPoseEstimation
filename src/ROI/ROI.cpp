#include <iostream>
#include <filesystem>
#include <thread>
#include <chrono>
#include "crop.hpp"
#include "model.hpp"
#include "debug.hpp"


const std::string FRAME_PATH = "C:\\temp\\webcam_frame.jpg";
bool DEBUG = false;
std::string msg;


int main(int argc, char* argv[]) {
	for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--verbose" || std::string(argv[i]) == "-v") {
        DEBUG = true;
		std::cout << "Debug mode enabled." << std::endl;
        }
    }

    // Preload onnx env.
    OrtContext ctx(L"resources/models/yolov8m-canned.onnx");

    std::filesystem::file_time_type lastProcessed;
    msg = "Waiting for frames at: " + FRAME_PATH;
    print_DEBUG(msg, DEBUG);


    // Main loop.
    while (true) {

        // Monitor any additional file changes for new images to analyse.
        if (std::filesystem::exists(FRAME_PATH)) {
            auto modTime = std::filesystem::last_write_time(FRAME_PATH);

            if (modTime != lastProcessed) {
                lastProcessed = modTime;

                cv::Mat image = loadImage_abs(FRAME_PATH);
                if (!image.empty()) {
                    cv::Mat letterboxed = letterboxResizeImage(image, 640, 640);
                    auto cropped = runInference(image, letterboxed, ctx, DEBUG);

                    // Save with timestamp to avoid overwriting
					//saveImage_ts(cropped, "cropped.jpg");
                    if (cropped.has_value()) {
                        saveImage_absOverwrite(cropped.value());
                    }
                    else {
						print_DEBUG("Failed detection.", DEBUG);
                    }
                }
                else {
					print_DEBUG("Failed to load image from: " + FRAME_PATH, DEBUG);
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}