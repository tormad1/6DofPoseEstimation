#include "pch.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>


cv::Mat loadImage(std::string imgName){
	
	std::string fullpath = "resources/images/source/" + imgName;
	cv::Mat image = cv::imread(fullpath);
	
	// Safety Check
	if (image.empty()){
		std::cout << "Could not read img" << std::endl;
		return cv::Mat();
	}

	return image;
}

cv::Mat loadImage_abs(const std::string& fullpath) {
	cv::Mat image = cv::imread(fullpath);
	return image;
}




// Memory load.
cv::Mat loadImage_memory(const unsigned char* jpegData, int jpegLength) {
	cv::Mat buf(1, jpegLength, CV_8UC1, (void*)jpegData);
	cv::Mat image = cv::imdecode(buf, cv::IMREAD_COLOR);
	return image;
}






// Custom path save.
void saveImage(cv::Mat image, std::string name) {

	std::string croppedName = "cropped_" + name;
	std::string fullpath = "resources/images/output/" + croppedName;
	cv::imwrite(fullpath, image);
	std::cout << "Image has been saved to: " << fullpath << std::endl;
}

void saveImage_abs(cv::Mat image, const std::string name) {
	std::string croppedName = "cropped_" + name;
	std::string absPath = "C:\\temp\\" + croppedName;
	cv::imwrite(absPath, image);
	std::cout << "Image has been saved to: " << absPath << std::endl;
}

void saveImage_absOverwrite(cv::Mat image) {
	cv::imwrite("C:\\temp\\cropped_tmp.jpg", image);

	// Same logic as in the C#
	// Save full file first then do quick overwrite.
	std::filesystem::copy_file(
		"C:\\temp\\cropped_tmp.jpg",
		"C:\\temp\\cropped.jpg",
		std::filesystem::copy_options::overwrite_existing
	);
}

void saveImage_ts(cv::Mat image, std::string name) {
	auto ts = std::chrono::system_clock::now().time_since_epoch().count();
	std::string timestampedName = "frame_" + std::to_string(ts) + ".jpg";
	saveImage_abs(image, timestampedName);
}


// Manually draw selection box as a test for i/o and shit
void selectROI(cv::Mat image) {
	cv::Rect2d roi = cv::selectROI(image);
	std::cout << "Selected ROI: " << roi << std::endl;
	cv::Mat croppedImage = image(roi);

	std::string outname = "manual-cropped.jpg";
	saveImage(croppedImage, outname);
}

// Fixed crop for testing.
void predeterminedCrop(cv::Mat image) {
	

	// Cropping is laid out with 4 coordinates
	// (x, y) is the top left corner of the box, and (width, height) is the size of the box.
	cv::Rect2d roi(100, 100, 200, 200);


	cv::Mat croppedImage = image(roi);
	// Using a Rect2d object on an image, crops it to the rects...
	// specified area and returns the cropped image as a new Mat object.


	std::string outname = "cropped.jpg";
	saveImage(croppedImage, outname);
}


// Resize image for model use.
cv::Mat letterboxResizeImage(const cv::Mat image, int width, int height) {
	int imgW = image.cols;
	int imgH = image.rows;

	float scale = std::min(
		static_cast<float>(width) / imgW,
		static_cast<float>(height) / imgH
	);

	int newImgW = static_cast<int>(std::round(imgW * scale));
	int newImgH = static_cast<int>(std::round(imgH * scale));

	int paddingX = (width - newImgW) / 2;
	int paddingY = (height - newImgH) / 2;

	cv::Mat resized;
	cv::resize(image, resized, cv::Size(newImgW, newImgH));

	// YOLO uses 114 as padding colour.
	cv::Mat output(height, width, image.type(), cv::Scalar(114, 114, 114));

	// Copy resized image into center of blank output image.
	resized.copyTo(
		output(cv::Rect(paddingX, paddingY, resized.cols, resized.rows))
	);

	return output;
}