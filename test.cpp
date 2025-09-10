#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img = cv::imread("image4.jpg");
    if (img.empty()) {
        std::cout << "Failed to load image" << std::endl;
        return -1;
    }
    std::cout << "Image size: " << img.cols << " x " << img.rows << std::endl;
    return 0;
}
