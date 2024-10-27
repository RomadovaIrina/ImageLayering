#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <fstream>


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

cv::Mat goToLAB(const cv::Mat& linImg) {
    cv::Mat labImg;
    cv::cvtColor(linImg, labImg, cv::COLOR_RGB2Lab);
    return labImg;
}


cv::Mat getAB(const cv::Mat& labImg) {
    cv::Mat abImg(labImg.size(), CV_32FC2);
    for (int row = 0; row < abImg.rows; row += 1) {
        for (int col = 0; col < abImg.cols; col += 1) {
            cv::Vec3f labPix = labImg.at<cv::Vec3f>(row, col);
            float a = labPix[1];
            float b = labPix[2];
            abImg.at<cv::Vec2f>(row, col) = cv::Vec2f(a, b);
        }
    }
    return abImg;
}

float getAngle(float x, float y) {
    float angle = std::atan2(y, x);
    if (angle < 0) {
        angle += 2 * CV_PI;
    }
    return angle;
}

std::vector<float> extractDataForHist(const cv::Mat& abImg) {
    std::vector<float> histData;
    for (int row = 0; row < abImg.rows; row += 1) {
        for (int col = 0; col < abImg.cols; col += 1) {
            cv::Vec2f abPix = abImg.at<cv::Vec2f>(row, col);
            float hue = getAngle(abPix[1], abPix[0]);
            histData.push_back(hue);
        }
    }
    return histData;
}


void exportCSV(const std::vector<float>& data) {
    std::ofstream file("outputFun.csv");
    if (file.is_open()) {
        for (float num : data) {
            file << num << "\n";
        }
        file.close();
        std::cout << "done";
    }
}

int main() {
    cv::Mat image = cv::imread("../test/funny.jpg");

    cv::Mat linearImage = make_Linear(image);

    cv::Mat normalizedImage;
    normalize(linearImage, normalizedImage, 0, 255, cv::NORM_MINMAX);
    normalizedImage.convertTo(normalizedImage, CV_8UC3);

    cv::Mat LAB = goToLAB(linearImage);

    cv::Mat normalizedLAB;
    normalize(LAB, normalizedLAB, 0, 255, cv::NORM_MINMAX);
    normalizedLAB.convertTo(normalizedLAB, CV_8UC3);

    imshow("Original image", image);
    imshow("Lin RGB image", normalizedImage);
    imshow("LAB", normalizedLAB);

    cv::waitKey(0);

    cv::Mat ab = getAB(LAB);
    std::vector<float> saturation = extractDataForHist(ab);
    exportCSV(saturation);
    return 0;
}
