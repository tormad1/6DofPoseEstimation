#include "crop.hpp"
#include "model.hpp"
#include <iostream>

int main() {

	initONNXRuntime();

	// Load the image
	std::string imageName = "monster-4.jpg";

	cv::Mat image = loadImage(imageName);
	cv::Mat letterboxed = letterboxResizeImage(image, 640, 640);


	cv::Mat cropped = runInference(image, letterboxed);
	saveImage(cropped, imageName);


	return 0;
}