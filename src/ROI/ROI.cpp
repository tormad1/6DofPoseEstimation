#include "crop.hpp"
#include <iostream>

int main() {

	// Load the image
	cv::Mat image = loadImage("monster.jpg");
	saveImage(image);

	return 0;
}