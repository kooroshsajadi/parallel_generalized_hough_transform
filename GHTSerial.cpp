#include <iostream>
#include <vector>
#include <utility>
#include <map>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>

typedef std::pair<int, int> Point;
typedef std::vector<Point> Model;
typedef std::map<std::tuple<int, int, double>, Point> RTable;

struct DetectionResult {
    int x;
    int y;
    double scale;
    int votes;
};

int calculateGradientDirection(const cv::Mat& image, int x, int y) {
    cv::Mat grad_x, grad_y;
    cv::Sobel(image, grad_x, CV_64F, 1, 0, 3);
    cv::Sobel(image, grad_y, CV_64F, 0, 1, 3);
    double dx = grad_x.at<double>(y, x);
    double dy = grad_y.at<double>(y, x);
    return std::atan2(dy, dx) * 180 / CV_PI;
}

int main() {
    cv::Mat templateImage = cv::imread("./resources/template_key.png", cv::IMREAD_GRAYSCALE);
    cv::Mat testImage = cv::imread("./resources/image_key.png", cv::IMREAD_GRAYSCALE);
    
    if (templateImage.empty() || testImage.empty()) {
        std::cerr << "Error: Could not load images." << std::endl;
        return -1;
    }
    
    Model model;
    std::vector<double> scales = {0.8, 1.0, 1.2};
    int numScales = scales.size();

    for (int y = 0; y < templateImage.rows; ++y) {
        for (int x = 0; x < templateImage.cols; ++x) {
            if (templateImage.at<uchar>(y, x) > 200) {
                model.push_back({x, y});
            }
        }
    }
    
    cv::Mat templateDisplay;
    cv::cvtColor(templateImage, templateDisplay, cv::COLOR_GRAY2BGR);
    for (const auto& pt : model) {
        cv::circle(templateDisplay, cv::Point(pt.first, pt.second), 2, cv::Scalar(0, 0, 255), -1);
    }
    
    RTable rTable;
    for (double scale: scales) {
        for (int i = 0; i < model.size(); ++i) {
            Point modelPoint = model[i];
            int gradientDirection = calculateGradientDirection(templateImage, modelPoint.first, modelPoint.second);
            Point scaledModelPoint = {static_cast<int>(modelPoint.first * scale), static_cast<int>(modelPoint.second * scale)};
            Point rVector = {-scaledModelPoint.first, -scaledModelPoint.second};
            rTable[std::make_tuple(gradientDirection, i, scale)] = rVector;
        }
    }
    
    int accumulatorWidth = testImage.cols;
    int accumulatorHeight = testImage.rows;
    std::vector<std::vector<std::vector<int>>> accumulator(accumulatorHeight, std::vector<std::vector<int>>(accumulatorWidth, std::vector<int>(numScales, 0)));
    
    cv::Mat testEdges;
    cv::Canny(testImage, testEdges, 100, 200); // Apply Canny edge detection.

    cv::imshow("Template Edges", testEdges);
    cv::waitKey(0);
    
    for (int y = 0; y < testEdges.rows; ++y) {
        for (int x = 0; x < testEdges.cols; ++x) {
            if (testEdges.at<uchar>(y, x) > 0) {
                int gradientDirection = calculateGradientDirection(testImage, x, y);
                for (double scale: scales) {
                    for (int i = 0; i < model.size(); ++i) {
                        auto key = std::make_tuple(gradientDirection, i, scale);
                        if (rTable.count(key)) {
                            Point rVector = rTable[key];
                            int refX = x - rVector.first;
                            int refY = y - rVector.second;
                            int scaleIndex = std::find(scales.begin(), scales.end(), scale) - scales.begin();
                            if (refX >= 0 && refX < accumulatorWidth && refY >= 0 && refY < accumulatorHeight && scaleIndex < numScales) {
                                accumulator[refY][refX][scaleIndex]++;
                            }
                        }
                    }
                }
            }
        }
    }
    
    std::vector<DetectionResult> detections;
    int voteThreshold = 50;
    for (int y = 0; y < accumulatorHeight; ++y) {
        for (int x = 0; x < accumulatorWidth; ++x) {
            for (int s = 0; s < numScales; ++s) {
                if (accumulator[y][x][s] > voteThreshold) {
                    detections.push_back({x, y, scales[s], accumulator[y][x][s]});
                }
            }
        }
    }
    
    cv::Mat testDisplay;
    cv::cvtColor(testImage, testDisplay, cv::COLOR_GRAY2BGR);
    for (const auto& detection : detections) {
        cv::circle(testDisplay, cv::Point(detection.x, detection.y), 10, cv::Scalar(255, 0, 0), 2);
    }
    
    cv::imshow("Template with Keypoints", templateDisplay);
    cv::imshow("Detected Objects", testDisplay);
    cv::waitKey(0);
    return 0;
}
