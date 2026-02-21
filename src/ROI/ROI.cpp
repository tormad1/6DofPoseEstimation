#include "crop.hpp"
#include "model.hpp"
#include <iostream>

int main() {

	// Load the image
	cv::Mat image = loadImage("monster.jpg");
	saveImage(image, "out.jpg");

	// Test YOLO model running.
	test_loadModel();

	//selectROI(image);
	predeterminedCrop(image);


	return 0;
}