#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <mpi.h>

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

// Function to calculate gradient direction
int calculateGradientDirection(const Mat &image, int x, int y) {
    Mat grad_x, grad_y;
    Sobel(image, grad_x, CV_32F, 1, 0, 3);
    Sobel(image, grad_y, CV_32F, 0, 1, 3);

    float dx = grad_x.at<float>(y, x);
    float dy = grad_y.at<float>(y, x);
    return static_cast<int>(atan2(dy, dx) * 180 / CV_PI) % ROTATION_LEVELS; // Quantize to 0-359 degrees
}

// Function to apply Canny edge detection
Mat applyCannyEdgeDetection(const Mat &image, int lowThreshold, int highThreshold) {
    Mat edges;
    Canny(image, edges, lowThreshold, highThreshold);
    return edges;
}

// Function to construct the R-Table
void constructRTable(const Mat &templ, RTable &rTable, Point reference) {
    for (int y = 0; y < templ.rows; y++) {
        for (int x = 0; x < templ.cols; x++) {
            // Check if the pixel is an edge pixel
            if (templ.at<uchar>(y, x) > 200) {
                // Calculate the gradient direction
                int angle = calculateGradientDirection(templ, x, y);

                // Quantize the angle into a bin
                int quantizedAngle = angle % ROTATION_LEVELS;

                // Compute the displacement vector from the edge pixel to the reference point
                Point vector = {reference.x - x, reference.y - y};

                // Add the vector to the R-Table for the quantized angle
                rTable[quantizedAngle].push_back(vector);
            }
        }
    }
}

// Function to detect objects using the R-Table (parallelized with MPI)
vector<Point> detectObjects(const Mat &edgeImage, const RTable &rTable, int voteThreshold, int dp, int minDistance, int rank, int size) {
    int width = edgeImage.cols / dp; // Scale accumulator width
    int height = edgeImage.rows / dp; // Scale accumulator height

    // Initialize the local accumulator space for each process
    Mat localAccumulator = Mat::zeros(height, width, CV_32SC1);

    // Divide the edge pixels among MPI processes
    int rowsPerProcess = edgeImage.rows / size;
    int startRow = rank * rowsPerProcess;
    int endRow = (rank == size - 1) ? edgeImage.rows : startRow + rowsPerProcess;

    // Iterate over the assigned rows of edge pixels
    for (int y = startRow; y < endRow; y++) {
        for (int x = 0; x < edgeImage.cols; x++) {
            if (edgeImage.at<uchar>(y, x) > 200) { // Check if it's an edge pixel
                // Calculate the gradient direction
                int angle = calculateGradientDirection(edgeImage, x, y);

                // Look up the corresponding vectors in the R-Table
                if (rTable.find(angle) != rTable.end()) {
                    for (const auto &vec : rTable.at(angle)) {
                        int cx = (x + vec.x) / dp; // Scale candidate center x
                        int cy = (y + vec.y) / dp; // Scale candidate center y

                        // Ensure the candidate center is within the accumulator bounds
                        if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
                            localAccumulator.at<int>(cy, cx)++; // Accumulate votes locally
                        }
                    }
                }
            }
        }
    }

    // Combine local accumulators from all processes into a global accumulator
    Mat globalAccumulator = Mat::zeros(height, width, CV_32SC1);
    MPI_Reduce(localAccumulator.data, globalAccumulator.data, width * height, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Only the root process (rank 0) performs peak detection and NMS
    vector<Point> detections;
    if (rank == 0) {
        // Find peaks in the global accumulator space
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (globalAccumulator.at<int>(y, x) > voteThreshold) {
                    detections.push_back(Point(x * dp, y * dp)); // Scale back to original image coordinates
                }
            }
        }

        // Non-maximum suppression (NMS) based on MIN_DISTANCE
        vector<Point> finalDetections;
        for (size_t i = 0; i < detections.size(); i++) {
            bool isMax = true;
            for (size_t j = 0; j < detections.size(); j++) {
                if (i != j && norm(detections[i] - detections[j]) < minDistance) {
                    if (globalAccumulator.at<int>(detections[i].y / dp, detections[i].x / dp) <
                        globalAccumulator.at<int>(detections[j].y / dp, detections[j].x / dp)) {
                        isMax = false;
                        break;
                    }
                }
            }
            if (isMax) {
                finalDetections.push_back(detections[i]);
            }
        }
        detections = finalDetections;
    }

    return detections;
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Load the template and convert it to grayscale (only rank 0)
    Mat templ;
    if (rank == 0) {
        templ = imread("resources/template_key.png", IMREAD_GRAYSCALE);
        if (templ.empty()) {
            cerr << "Error: Could not load template image." << endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    // Broadcast the template to all processes
    int templRows, templCols;
    if (rank == 0) {
        templRows = templ.rows;
        templCols = templ.cols;
    }
    MPI_Bcast(&templRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&templCols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        templ = Mat(templRows, templCols, CV_8UC1);
    }
    MPI_Bcast(templ.data, templRows * templCols, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Load the input image (only rank 0)
    Mat coloredImage;
    if (rank == 0) {
        coloredImage = imread("resources/image_key.png");
        if (coloredImage.empty()) {
            cerr << "Error: Could not load input image." << endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    // Broadcast the input image to all processes.
    int imageRows, imageCols;
    if (rank == 0) {
        imageRows = coloredImage.rows;
        imageCols = coloredImage.cols;
    }
    MPI_Bcast(&imageRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imageCols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        coloredImage = Mat(imageRows, imageCols, CV_8UC3);
    }
    MPI_Bcast(coloredImage.data, imageRows * imageCols * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Create grayscale image.
    Mat image;
    cvtColor(coloredImage, image, COLOR_RGB2GRAY);

    // Apply Canny edge detection to the template and input image.
    Mat edgeTemplate = applyCannyEdgeDetection(templ, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD);
    Mat edgeImage = applyCannyEdgeDetection(image, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD);

    // Define the reference point (center of the template).
    Point reference = {templ.cols / 2, templ.rows / 2};

    // Construct the R-Table (only rank 0).
    RTable rTable;
    if (rank == 0) {
        constructRTable(edgeTemplate, rTable, reference);
    }

    // Broadcast the R-Table to all processes.
    int rTableSize;
    if (rank == 0) {
        rTableSize = rTable.size();
    }
    MPI_Bcast(&rTableSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < rTableSize; i++) {
        int angle, vectorCount;
        if (rank == 0) {
            auto it = rTable.begin();
            advance(it, i);
            angle = it->first;
            vectorCount = it->second.size();
        }
        MPI_Bcast(&angle, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&vectorCount, 1, MPI_INT, 0, MPI_COMM_WORLD);

        vector<Point> vectors(vectorCount);
        if (rank == 0) {
            vectors = rTable[angle];
        }
        MPI_Bcast(vectors.data(), vectorCount * 2, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            rTable[angle] = vectors;
        }
    }

    // Detect objects in the input image using the R-Table.
    vector<Point> detections = detectObjects(edgeImage, rTable, VOTE_THRESHOLD, ACCUMULATOR_DEPTH, MIN_DISTANCE, rank, size);

    // Only rank 0 draws the detected objects and displays the result.
    if (rank == 0) {
        // Draw the detected objects on the input image.
        for (const auto &center : detections) {
            // Draw a circle at the detected center.
            circle(coloredImage, center, 10, Scalar(255, 0, 0), 2);

            // Draw a bounding box around the detected object.
            Rect boundingBox(center.x - templ.cols / 2, center.y - templ.rows / 2, templ.cols, templ.rows);
            rectangle(coloredImage, boundingBox, Scalar(0, 255, 0), 2);
        }

        imshow("Detected Objects", coloredImage);
        waitKey(0);
    }

    // Finalize MPI.
    MPI_Finalize();
    return EXIT_SUCCESS;
}