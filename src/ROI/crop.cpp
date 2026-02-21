#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat loadImage(std::string imgName){
	
	std::string fullpath = "resources/images/" + imgName;
	cv::Mat image = cv::imread(fullpath);
	
	// Safety Check
	if (image.empty()){
		std::cout << "Could not read img" << std::endl;
		return cv::Mat();
	}

	return image;
}


void saveImage(cv::Mat image) {

	cv::imwrite("resources/images/cropped.jpg", image);
	std::cout << "Image has been saved" << std::endl;
}