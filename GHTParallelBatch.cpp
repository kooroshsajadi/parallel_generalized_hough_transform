#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <mpi.h>
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

vector<Point> detectObjects(const Mat &edgeImage, const RTable &rTable, int voteThreshold, int dp, int minDistance, int rank, int size) {
    int width = edgeImage.cols / dp;
    int height = edgeImage.rows / dp;

    Mat localAccumulator = Mat::zeros(height, width, CV_32SC1);

    int rowsPerProcess = edgeImage.rows / size;
    int startRow = rank * rowsPerProcess;
    int endRow = (rank == size - 1) ? edgeImage.rows : startRow + rowsPerProcess;

    for (int y = startRow; y < endRow; y++) {
        for (int x = 0; x < edgeImage.cols; x++) {
            if (edgeImage.at<uchar>(y, x) > 200) {
                int angle = calculateGradientDirection(edgeImage, x, y);
                if (rTable.find(angle) != rTable.end()) {
                    for (const auto &vec : rTable.at(angle)) {
                        int cx = (x + vec.x) / dp;
                        int cy = (y + vec.y) / dp;
                        if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
                            localAccumulator.at<int>(cy, cx)++;
                        }
                    }
                }
            }
        }
    }

    Mat globalAccumulator = Mat::zeros(height, width, CV_32SC1);
    MPI_Reduce(localAccumulator.data, globalAccumulator.data, width * height, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    vector<Point> detections;
    if (rank == 0) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (globalAccumulator.at<int>(y, x) > voteThreshold) {
                    detections.push_back(Point(x * dp, y * dp));
                }
            }
        }

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
    MPI_Init(&argc, &argv);

    // Start timing the entire batch process
    auto start = chrono::high_resolution_clock::now();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Print the rank of the current process
    cout << "Process rank: " << rank << " out of " << size << " total processes" << endl;

    // Load and broadcast the template.
    Mat templ;
    if (rank == 0) {
        templ = imread("resources/template_key.png", IMREAD_GRAYSCALE);
        if (templ.empty()) {
            cerr << "Error: Could not load template image." << endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

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

    // Construct and broadcast the R-Table.
    RTable rTable;
    if (rank == 0) {
        Mat edgeTemplate = applyCannyEdgeDetection(templ, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD);
        Point reference = {templ.cols / 2, templ.rows / 2};
        constructRTable(edgeTemplate, rTable, reference);
    }

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

    // Load all image filenames on rank 0 and broadcast
    vector<String> imageFiles;
    if (rank == 0) {
        glob("resources/dataset/*.png", imageFiles, false);
        if (imageFiles.empty()) {
            cerr << "Error: No PNG images found in resources/dataset directory." << endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        cout << "Processing " << imageFiles.size() << " images from resources/dataset/" << endl;
    }

    int numImages = (rank == 0) ? static_cast<int>(imageFiles.size()) : 0;
    MPI_Bcast(&numImages, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Distribute images across processes
    int imagesPerProcess = numImages / size;
    int startIdx = rank * imagesPerProcess;
    int endIdx = (rank == size - 1) ? numImages : startIdx + imagesPerProcess;

    vector<Mat> localImages(endIdx - startIdx);
    vector<vector<Point>> localDetections(endIdx - startIdx);

    // Rank 0 loads all images and sends to respective processes
    if (rank == 0) {
        vector<Mat> allImages(numImages);
        for (int i = 0; i < numImages; i++) {
            allImages[i] = imread(imageFiles[i]);
            if (allImages[i].empty()) {
                cerr << "Warning: Could not load image " << imageFiles[i] << endl;
                allImages[i] = Mat(); // Empty placeholder
            }
        }

        // Send images to other processes.
        for (int r = 0; r < size; r++) {
            int rStart = r * imagesPerProcess;
            int rEnd = (r == size - 1) ? numImages : rStart + imagesPerProcess;
            for (int i = rStart; i < rEnd; i++) {
                int rows = allImages[i].rows;
                int cols = allImages[i].cols;
                int type = allImages[i].type();
                if (r == 0) {
                    localImages[i - rStart] = allImages[i];
                } else {
                    MPI_Send(&rows, 1, MPI_INT, r, i, MPI_COMM_WORLD);
                    MPI_Send(&cols, 1, MPI_INT, r, i, MPI_COMM_WORLD);
                    MPI_Send(&type, 1, MPI_INT, r, i, MPI_COMM_WORLD);
                    if (!allImages[i].empty()) {
                        MPI_Send(allImages[i].data, rows * cols * allImages[i].channels(), MPI_UNSIGNED_CHAR, r, i, MPI_COMM_WORLD);
                    }
                }
            }
        }
    } else {
        // Receive images assigned to this process.
        for (int i = startIdx; i < endIdx; i++) {
            int rows, cols, type;
            MPI_Recv(&rows, 1, MPI_INT, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&cols, 1, MPI_INT, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&type, 1, MPI_INT, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            localImages[i - startIdx] = Mat(rows, cols, type);
            if (rows > 0 && cols > 0) {
                MPI_Recv(localImages[i - startIdx].data, rows * cols * localImages[i - startIdx].channels(), MPI_UNSIGNED_CHAR, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    // Process local batch of images.
    for (int i = 0; i < localImages.size(); i++) {
        if (!localImages[i].empty()) {
            cout << "Process rank: " << rank << " Processing local image #" << i << " ..." << endl;
            Mat image;
            cvtColor(localImages[i], image, COLOR_RGB2GRAY);
            Mat edgeImage = applyCannyEdgeDetection(image, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD);
            localDetections[i] = detectObjects(edgeImage, rTable, VOTE_THRESHOLD, ACCUMULATOR_DEPTH, MIN_DISTANCE, rank, size);
        }
    }

    // Gather detections to rank 0 (simplified: assuming small number of detections)
    vector<vector<Point>> allDetections(numImages);
    if (rank == 0) {
        for (int i = 0; i < localDetections.size(); i++) {
            allDetections[i] = localDetections[i];
        }
        for (int r = 1; r < size; r++) {
            int rStart = r * imagesPerProcess;
            int rEnd = (r == size - 1) ? numImages : rStart + imagesPerProcess;
            for (int i = rStart; i < rEnd; i++) {
                int detectionSize;
                MPI_Recv(&detectionSize, 1, MPI_INT, r, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                allDetections[i].resize(detectionSize);
                if (detectionSize > 0) {
                    MPI_Recv(allDetections[i].data(), detectionSize * sizeof(Point), MPI_BYTE, r, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
    } else {
        for (int i = 0; i < localDetections.size(); i++) {
            int detectionSize = localDetections[i].size();
            MPI_Send(&detectionSize, 1, MPI_INT, 0, startIdx + i, MPI_COMM_WORLD);
            if (detectionSize > 0) {
                MPI_Send(localDetections[i].data(), detectionSize * sizeof(Point), MPI_BYTE, 0, startIdx + i, MPI_COMM_WORLD);
            }
        }
    }

    // End timing
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    // Rank 0 processes results
    if (rank == 0) {
        cout << "Parallel GHT batch processing time for " << numImages << " images: " << duration.count() << " seconds" << endl;

        // Load images again for drawing (simplification; could optimize by storing)
        vector<Mat> coloredImages(numImages);
        for (int i = 0; i < numImages; i++) {
            coloredImages[i] = imread(imageFiles[i]);
            if (!coloredImages[i].empty()) {
                for (const auto center : allDetections[i]) {
                    circle(coloredImages[i], center, 10, Scalar(255, 0, 0), 2);
                    Rect boundingBox(center.x - templ.cols / 2, center.y - templ.rows / 2, templ.cols, templ.rows);
                    rectangle(coloredImages[i], boundingBox, Scalar(0, 255, 0), 2);
                }

                // imshow("Detected Objects - " + imageFiles[i], coloredImages[i]);
                // waitKey(0);
                
                // string outputFile = "resources/dataset/output_" + to_string(i) + ".png";
                // imwrite(outputFile, coloredImages[i]);
            }
        }
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}