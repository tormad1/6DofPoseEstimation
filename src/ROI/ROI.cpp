#include "crop.hpp"
#include "model.hpp"
#include <iostream>

int main() {

	// Load the image
	cv::Mat image = loadImage("monster.jpg");
	cv::Mat resizedImage = letterboxResizeImage(image, 640, 640);
	saveImage(resizedImage, "resized.jpg");

	predeterminedCrop(resizedImage);


	return 0;
}