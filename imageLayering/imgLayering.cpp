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

cv::Vec3f RGBtoLAB(cv::Vec3f& linRGB) {
    float b = linRGB[0];
    float g = linRGB[1];
    float r = linRGB[2];
    
    float L = (r-g) / sqrt(2);
    float alpha = (2*b-r-g) / sqrt(6);
    float beta = (r+g+b) / sqrt(3);
    return cv::Vec3f(L, alpha, beta);
}

cv::Mat consvertLAB(const cv::Mat& linImg) {

    cv::Mat labImg(linImg.size(), CV_32FC3);

    for (int y = 0; y < linImg.rows; ++y) {
        for (int x = 0; x < linImg.cols; ++x) {
            cv::Vec3f linRGB = linImg.at<cv::Vec3f>(y, x);
            labImg.at<cv::Vec3f>(y, x) = RGBtoLAB(linRGB);
        }
    }

    return labImg;
}

cv::Mat goToLAB(const cv::Mat& linImg) {
    cv::Mat labImg;
    cv::cvtColor(linImg, labImg, cv::COLOR_BGR2Lab);
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

void displayABProjection(const cv::Mat& abImg) {
    cv::Mat displayImg(abImg.rows, abImg.rows, CV_8UC3, cv::Scalar(128, 128, 128));

    cv::line(displayImg, cv::Point(displayImg.cols / 2, 0), cv::Point(displayImg.cols / 2, displayImg.rows), cv::Scalar(0, 0, 0), 1);
    cv::line(displayImg, cv::Point(0, displayImg.rows / 2), cv::Point(displayImg.cols, displayImg.rows / 2), cv::Scalar(0, 0, 0), 1);

    for (int row = 0; row < abImg.rows; row++) {
        for (int col = 0; col < abImg.cols; col++) {
            cv::Vec2f abPix = abImg.at<cv::Vec2f>(row, col);
            int a = static_cast<int>((abPix[0] + 128) * 2);
            int b = static_cast<int>((abPix[1] + 128) * 2); 
            if (a >= 0 && a < displayImg.cols && b >= 0 && b < displayImg.rows) {
                int intensity = static_cast<int>(abPix[0] + abPix[1]) % 255;
                displayImg.at<cv::Vec3b>(b, a) = cv::Vec3b(intensity, intensity, intensity);
            }
        }
    }
    cv::imshow("AB Projection", displayImg);
    cv::waitKey(0);
}


void exportCSV(const std::vector<float>& data) {
    std::ofstream file("outputEGE.csv");
    if (file.is_open()) {
        for (float num : data) {
            file << num << "\n";
        }
        file.close();
        std::cout << "done";
    }
}

int main() {
    cv::Mat image = cv::imread("../test/temp.jpg");

    {
        cv::Mat linearImage = make_Linear(image);

        cv::Mat normalizedImage;
        normalize(linearImage, normalizedImage, 0, 255, cv::NORM_MINMAX);
        normalizedImage.convertTo(normalizedImage, CV_8UC3);

        cv::Mat LAB = consvertLAB(linearImage);

        cv::Mat normalizedLAB;
        normalize(LAB, normalizedLAB, 0, 255, cv::NORM_MINMAX);
        //normalizedLAB.convertTo(normalizedLAB, CV_8UC3);
        normalizedLAB.convertTo(normalizedLAB, CV_8UC3);

        imshow("Original image", image);
        imshow("Lin RGB image", normalizedImage);
        imshow("LAB", normalizedLAB);
    }


    cv::Mat linearImage = make_Linear(image);
    cv::Mat LAB_2 = goToLAB(linearImage);
    cv::Mat ab = getAB(LAB_2);

    cv::Mat normalizedLAB_2;
    normalize(LAB_2, normalizedLAB_2, 0, 255, cv::NORM_MINMAX);
    normalizedLAB_2.convertTo(normalizedLAB_2, CV_8UC3);
    imshow("LAB_2", normalizedLAB_2);

    displayABProjection(ab);
    std::vector<float> saturation = extractDataForHist(ab);
    exportCSV(saturation);

    return 0;
}
