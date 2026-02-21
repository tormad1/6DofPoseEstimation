#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img = cv::imread("monster-in-hand.jpg");
    if (img.empty()) {
        std::cerr << "Failed to load test.jpg\n";
        return 1;
    }
    cv::imwrite("out.jpg", img);
    std::cout << "Wrote out.jpg\n";
    return 0;
}