#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "crop.hpp"
#include "model.hpp"

void test_allMonsters() {
	for (int i = 1; i <= 8; i++) {
		std::string imgName = "monster-" + std::to_string(i) + ".jpg";
		cv::Mat image = loadImage(imgName);
		cv::Mat letterboxed = letterboxResizeImage(image, 640, 640);
		cv::Mat cropped = runInference(image, letterboxed);
		saveImage(cropped, imgName);

		std::cout << std::to_string(i) + " done." << std::endl;
		std::cout << "Hello world" << std::endl;
	}
}