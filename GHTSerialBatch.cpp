#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace cv;
using namespace std;

// ==================================================
// Hyperparameters
// ==================================================
const int ROTATION_LEVELS = 360;       // Number of rotation levels (1-degree resolution)
const int ACCUMULATOR_DEPTH = 2;       // Resolution of the accumulator (scaling factor)
const int CANNY_LOW_THRESHOLD = 30;    // Canny edge detection low threshold
const int CANNY_HIGH_THRESHOLD = 110;  // Canny edge detection high threshold
const int VOTE_THRESHOLD = 40;         // Accumulator threshold for object detection
const int MIN_DISTANCE = 80;           // Minimum distance between detected objects
// ==================================================

typedef map<int, vector<Point>> RTable; // R-Table: maps quantized angles to displacement vectors

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

vector<Point> detectObjects(const Mat &edgeImage, const RTable &rTable, int voteThreshold, int dp, int minDistance) {
    int width = edgeImage.cols / dp;
    int height = edgeImage.rows / dp;

    Mat accumulator = Mat::zeros(height, width, CV_32SC1);

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

    return finalDetections;
}

int main() {
    // Start timing the entire batch process
    auto start = chrono::high_resolution_clock::now();

    // Load the template and convert it to grayscale
    Mat templ = imread("resources/template_key.png", IMREAD_GRAYSCALE);
    if (templ.empty()) {
        cerr << "Error: Could not load template image." << endl;
        return EXIT_FAILURE;
    }

    // Apply Canny edge detection to the template
    Mat edgeTemplate = applyCannyEdgeDetection(templ, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD);

    // Define the reference point (center of the template)
    Point reference = {templ.cols / 2, templ.rows / 2};

    // Construct the R-Table once for all images
    RTable rTable;
    constructRTable(edgeTemplate, rTable, reference);

    // Read all .png files from the dataset directory
    vector<String> imageFiles;
    glob("resources/dataset/*.png", imageFiles, false);
    if (imageFiles.empty()) {
        cerr << "Error: No PNG images found in resources/dataset directory." << endl;
        return EXIT_FAILURE;
    }

    cout << "Processing " << imageFiles.size() << " images from resources/dataset/" << endl;

    // Process each image in the batch
    vector<Mat> coloredImages(imageFiles.size());
    vector<vector<Point>> allDetections(imageFiles.size());

    for (size_t i = 0; i < imageFiles.size(); i++) {
        // Load the input image
        coloredImages[i] = imread(imageFiles[i]);
        if (coloredImages[i].empty()) {
            cerr << "Error: Could not load image " << imageFiles[i] << endl;
            continue;
        }

        cout << "Processing image #" << i << " ..." << endl;

        // Create grayscale image.
        Mat image;
        cvtColor(coloredImages[i], image, COLOR_RGB2GRAY);

        // Apply Canny edge detection
        Mat edgeImage = applyCannyEdgeDetection(image, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD);

        // Detect objects using the R-Table
        allDetections[i] = detectObjects(edgeImage, rTable, VOTE_THRESHOLD, ACCUMULATOR_DEPTH, MIN_DISTANCE);
    }

    // Stop timing.
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    // Print execution time for the batch
    cout << "Execution time for batch processing " << imageFiles.size() << " images: " << duration.count() << " seconds" << endl;

    // Draw detections on all images and save.
    for (size_t i = 0; i < imageFiles.size(); i++) {
        if (!coloredImages[i].empty()) {
            for (const auto center : allDetections[i]) { // Fixed typo: '¢er' to 'center'
                circle(coloredImages[i], center, 10, Scalar(255, 0, 0), 2);
                Rect boundingBox(center.x - templ.cols / 2, center.y - templ.rows / 2, templ.cols, templ.rows);
                rectangle(coloredImages[i], boundingBox, Scalar(0, 255, 0), 2);
            }

            // Uncomment to display each image
            // imshow("Detected Objects - " + imageFiles[i], coloredImages[i]);
            // waitKey(0);

            // Optionally save the output
            // string outputFile = "resources/dataset/output_" + to_string(i) + ".png";
            // imwrite(outputFile, coloredImages[i]);
        }
    }

    return EXIT_SUCCESS;
}