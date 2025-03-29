#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace cv;
using namespace std;

// Hyperparameters
const int ROTATION_LEVELS = 360;
const int ACCUMULATOR_DEPTH = 2;
const int CANNY_LOW_THRESHOLD = 30;
const int CANNY_HIGH_THRESHOLD = 110;
const int VOTE_THRESHOLD = 40;
const int MIN_DISTANCE = 80;

typedef map<int, vector<Point>> RTable;

int calculateGradientDirection(const Mat &image, int x, int y) {
    Mat grad_x, grad_y;
    Sobel(image, grad_x, CV_32F, 1, 0, 3);
    Sobel(image, grad_y, CV_32F, 0, 1, 3);

    float dx = grad_x.at<float>(y, x);
    float dy = grad_y.at<float>(y, x);
    return static_cast<int>(atan2(dy, dx) * 180 / CV_PI) % ROTATION_LEVELS;
}

Mat applyCannyEdgeDetection(const Mat &image, int lowThreshold, int highThreshold) {
    Mat edges;
    Canny(image, edges, lowThreshold, highThreshold);
    return edges;
}

void constructRTable(const Mat &templ, RTable &rTable, Point reference) {
    for (int y = 0; y < templ.rows; y++) {
        for (int x = 0; x < templ.cols; x++) {
            if (templ.at<uchar>(y, x) > 200) {
                int angle = calculateGradientDirection(templ, x, y);
                int quantizedAngle = angle % ROTATION_LEVELS;
                Point vector = {reference.x - x, reference.y - y};
                rTable[quantizedAngle].push_back(vector);
            }
        }
    }
}

vector<Point> detectObjects(const Mat &edgeImage, const RTable &rTable, int voteThreshold, int dp, int minDistance, double &voting_time, double &peak_time) {
    int width = edgeImage.cols / dp;
    int height = edgeImage.rows / dp;

    Mat accumulator = Mat::zeros(height, width, CV_32SC1);

    // Voting phase
    auto voting_start = chrono::high_resolution_clock::now();
    for (int y = 0; y < edgeImage.rows; y++) {
        for (int x = 0; x < edgeImage.cols; x++) {
            if (edgeImage.at<uchar>(y, x) > 200) {
                int angle = calculateGradientDirection(edgeImage, x, y);
                if (rTable.find(angle) != rTable.end()) {
                    for (const auto &vec : rTable.at(angle)) {
                        int cx = (x + vec.x) / dp;
                        int cy = (y + vec.y) / dp;
                        if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
                            accumulator.at<int>(cy, cx)++;
                        }
                    }
                }
            }
        }
    }
    auto voting_end = chrono::high_resolution_clock::now();
    voting_time = chrono::duration<double>(voting_end - voting_start).count();

    // Peak detection and NMS phase
    auto peak_start = chrono::high_resolution_clock::now();
    vector<Point> detections;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (accumulator.at<int>(y, x) > voteThreshold) {
                detections.push_back(Point(x * dp, y * dp));
            }
        }
    }

    vector<Point> finalDetections;
    for (size_t i = 0; i < detections.size(); i++) {
        bool isMax = true;
        for (size_t j = 0; j < detections.size(); j++) {
            if (i != j && norm(detections[i] - detections[j]) < minDistance) {
                if (accumulator.at<int>(detections[i].y / dp, detections[i].x / dp) <
                    accumulator.at<int>(detections[j].y / dp, detections[j].x / dp)) {
                    isMax = false;
                    break;
                }
            }
        }
        if (isMax) {
            finalDetections.push_back(detections[i]);
        }
    }
    auto peak_end = chrono::high_resolution_clock::now();
    peak_time = chrono::duration<double>(peak_end - peak_start).count();

    return finalDetections;
}

int main() {
    auto total_start = chrono::high_resolution_clock::now();

    // Image loading and grayscale conversion (template)
    auto load_start = chrono::high_resolution_clock::now();
    Mat templ = imread("resources/template_key.png", IMREAD_GRAYSCALE);
    if (templ.empty()) {
        cerr << "Error: Could not load template image." << endl;
        return EXIT_FAILURE;
    }
    auto load_end = chrono::high_resolution_clock::now();
    double load_time = chrono::duration<double>(load_end - load_start).count();

    // Canny edge detection on template
    auto canny_templ_start = chrono::high_resolution_clock::now();
    Mat edgeTemplate = applyCannyEdgeDetection(templ, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD);
    auto canny_templ_end = chrono::high_resolution_clock::now();
    double canny_templ_time = chrono::duration<double>(canny_templ_end - canny_templ_start).count();

    // R-Table construction
    auto rtable_start = chrono::high_resolution_clock::now();
    Point reference = {templ.cols / 2, templ.rows / 2};
    RTable rTable;
    constructRTable(edgeTemplate, rTable, reference);
    auto rtable_end = chrono::high_resolution_clock::now();
    double rtable_time = chrono::duration<double>(rtable_end - rtable_start).count();

    // Load dataset images
    vector<String> imageFiles;
    glob("resources/dataset/*.png", imageFiles, false);
    if (imageFiles.empty()) {
        cerr << "Error: No PNG images found in resources/dataset directory." << endl;
        return EXIT_FAILURE;
    }

    cout << "Processing " << imageFiles.size() << " images from resources/dataset/" << endl;

    vector<Mat> coloredImages(imageFiles.size());
    vector<vector<Point>> allDetections(imageFiles.size());

    double total_load_time = load_time; // Include template loading
    double total_canny_image_time = 0;
    double total_voting_time = 0;
    double total_peak_time = 0;
    double total_visualize_time = 0;

    for (size_t i = 0; i < imageFiles.size(); i++) {
        // Load and convert image
        auto img_load_start = chrono::high_resolution_clock::now();
        coloredImages[i] = imread(imageFiles[i]);
        if (coloredImages[i].empty()) {
            cerr << "Error: Could not load image " << imageFiles[i] << endl;
            continue;
        }
        Mat image;
        cvtColor(coloredImages[i], image, COLOR_RGB2GRAY);
        auto img_load_end = chrono::high_resolution_clock::now();
        total_load_time += chrono::duration<double>(img_load_end - img_load_start).count();

        cout << "Processing image #" << i << " ..." << endl;

        // Canny edge detection on image
        auto canny_start = chrono::high_resolution_clock::now();
        Mat edgeImage = applyCannyEdgeDetection(image, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD);
        auto canny_end = chrono::high_resolution_clock::now();
        total_canny_image_time += chrono::duration<double>(canny_end - canny_start).count();

        // Detect objects
        double voting_time = 0, peak_time = 0;
        allDetections[i] = detectObjects(edgeImage, rTable, VOTE_THRESHOLD, ACCUMULATOR_DEPTH, MIN_DISTANCE, voting_time, peak_time);
        total_voting_time += voting_time;
        total_peak_time += peak_time;

        // Visualization
        auto vis_start = chrono::high_resolution_clock::now();
        for (const auto center : allDetections[i]) {
            circle(coloredImages[i], center, 10, Scalar(255, 0, 0), 2);
            Rect boundingBox(center.x - templ.cols / 2, center.y - templ.rows / 2, templ.cols, templ.rows);
            rectangle(coloredImages[i], boundingBox, Scalar(0, 255, 0), 2);
        }
        imshow("Detected Objects - " + imageFiles[i], coloredImages[i]);
        // waitKey(0);
        string outputFile = "resources/dataset/output_" + to_string(i) + ".png";
        imwrite(outputFile, coloredImages[i]);
        auto vis_end = chrono::high_resolution_clock::now();
        total_visualize_time += chrono::duration<double>(vis_end - vis_start).count();
    }

    auto total_end = chrono::high_resolution_clock::now();
    double total_time = chrono::duration<double>(total_end - total_start).count();

    // Print timing results
    cout << "Total execution time: " << total_time << " seconds" << endl;
    cout << "Image Loading & Grayscale time: " << total_load_time << " seconds" << endl;
    cout << "Canny Edge Detection (Template): " << canny_templ_time << " seconds" << endl;
    cout << "Canny Edge Detection (Images): " << total_canny_image_time << " seconds" << endl;
    cout << "R-Table Construction: " << rtable_time << " seconds" << endl;
    cout << "Voting time: " << total_voting_time << " seconds" << endl;
    cout << "Peak Detection & NMS time: " << total_peak_time << " seconds" << endl;
    cout << "Visualization: " << total_visualize_time << " seconds" << endl;

    return EXIT_SUCCESS;
}