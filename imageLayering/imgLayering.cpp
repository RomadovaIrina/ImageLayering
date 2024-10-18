#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


cv::Vec3f convert_pix(cv::Vec3b& pix) {
    double edge = 0.040449936;
    double a = 0.055;
    double x = 2.4;

    cv::Vec3f lin;
    for (int i = 0; i < 3; ++i) {
        double srgb_pix = pix[i] / 255.0f;
        if (srgb_pix <= edge) {
            lin[i] = srgb_pix / 12.92f;
        }
        else {
            lin[i] = std::pow((srgb_pix + a) / (1 + a), x);
        }
    }

    return lin;
}


cv::Mat make_Linear(const cv::Mat& img) {

    cv::Mat linear(img.size(), CV_32FC3);

    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            cv::Vec3b srgbPixel = img.at<cv::Vec3b>(y, x);
            linear.at<cv::Vec3f>(y, x) = convert_pix(srgbPixel);
        }
    }

    return linear;

}


int main() {
    cv::Mat image = cv::imread("../test/2.png");

    cv::Mat linearImage = make_Linear(image);

    cv::Mat normalizedImage;
    normalize(linearImage, normalizedImage, 0, 255, cv::NORM_MINMAX);
    normalizedImage.convertTo(normalizedImage, CV_8UC3);


    imshow("Original image", image);
    imshow("Lin RGB image", normalizedImage);

    cv::waitKey(0);
    return 0;
}
