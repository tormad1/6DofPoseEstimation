#pragma once
#include <opencv2/opencv.hpp>

cv::Mat loadImage(std::string imgName);
void saveImage(cv::Mat image, std::string name);
void selectROI(cv::Mat image);
void predeterminedCrop(cv::Mat image);