#pragma once
#include <opencv2/opencv.hpp>

cv::Mat loadImage(std::string imgName);
cv::Mat loadImage_abs(const std::string& fullpath);
cv::Mat loadImage_memory(const unsigned char* jpegData, int jpegLength);

void saveImage(cv::Mat image, std::string name);
void saveImage_abs(cv::Mat image, const std::string name);
void saveImage_ts(cv::Mat image, std::string name);
void saveImage_absOverwrite(cv::Mat image);

void selectROI(cv::Mat image);
void predeterminedCrop(cv::Mat image);
cv::Mat letterboxResizeImage(cv::Mat image, int width, int height);