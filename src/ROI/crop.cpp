#include <opencv2/opencv.hpp>
#include <iostream>

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

// Custom path save.
void saveImage(cv::Mat image, std::string name) {

	std::string croppedName = "cropped-" + name;
	std::string fullpath = "resources/images/output/" + croppedName;
	cv::imwrite(fullpath, image);
	std::cout << "Image has been saved to: " << fullpath << std::endl;
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