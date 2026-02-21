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

// Save feature with custom name.
void saveImage(cv::Mat image, std::string name) {

	std::string fullpath = "resources/images/" + name;
	cv::imwrite(fullpath, image);
	std::cout << "Image has been saved" << std::endl;
}


// Manually draw selection box as a test for i/o and shit
void selectROI(cv::Mat image) {
	cv::Rect2d roi = cv::selectROI(image);
	std::cout << "Selected ROI: " << roi << std::endl;
	cv::Mat croppedImage = image(roi);

	std::string outname = "cropped.jpg";
	saveImage(croppedImage, outname);
}

// Fixed crop for testing.
void predeterminedCrop(cv::Mat image) {
	
	cv::Rect2d roi(100, 100, 200, 200);
	std::cout << "Selected ROI: " << roi << std::endl;
	cv::Mat croppedImage = image(roi);
	std::string outname = "cropped.jpg";
	saveImage(croppedImage, outname);
}