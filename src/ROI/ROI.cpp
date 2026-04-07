#include "crop.hpp"
#include "model.hpp"
#include <iostream>

int main() {

	// Load the image
	cv::Mat image = loadImage("bottle.jpg");
	cv::Mat resizedImage = letterboxResizeImage(image, 640, 640);
	saveImage(resizedImage, "resized.jpg");

	runInference(resizedImage);

	return 0;
}