#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
using namespace cv;
using namespace std;

int main() {
    // Load source image and grayscale template.
    Mat image = imread("resources/image_key.png");
    Mat templ = imread("resources/template_key.png", IMREAD_GRAYSCALE);

    // Create grayscale image.
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_RGB2GRAY); // Grayscaling makes edge detection more effective.

    // Create variable for location, scale and rotation of detected templates.
    vector<Vec4f> positionBallard;

    // Set template's width and height.
    int w = templ.cols;
    int h = templ.rows;

    // Create ballard and set options.
    Ptr<GeneralizedHoughBallard> ballard = createGeneralizedHoughBallard();
    ballard->setMinDist(50); // Minimum distance between detected objects in pixels
    ballard->setLevels(360); // Set the number of rotation levels; 360 is maximal value.
    ballard->setDp(2); // Resolution of the accumulator used to detect centers of the objects
    ballard->setMaxBufferSize(1000); // Maximal size of inner buffers
    ballard->setVotesThreshold(40); // Accumulator threshold parameter (threshold for object detection)
    ballard->setCannyLowThresh(30); // The first threshold for the hysteresis procedure in Canny edge detector
    ballard->setCannyHighThresh(110); // The second threshold for the hysteresis procedure in Canny edge detector
    ballard->setTemplate(templ);

    // Execute Ballard detection.
    cout << "Executing Ballard detection..." << endl;
    ballard->detect(grayImage, positionBallard);
    cout << "Ballard detection completed. Found " << positionBallard.size() << " objects." << endl;

    //  draw Ballard.
    for (vector<Vec4f>::iterator iter = positionBallard.begin(); iter != positionBallard.end(); ++iter) {
        RotatedRect rRect = RotatedRect(Point2f((*iter)[0], (*iter)[1]), Size2f(w * (*iter)[2], h * (*iter)[2]), (*iter)[3]);
        Point2f vertices[4];
        rRect.points(vertices);
        for (int i = 0; i < 4; i++)
            line(image, vertices[i], vertices[(i + 1) % 4], Scalar(255, 0, 0), 6);
    }

    imshow("result_img", image);
    waitKey();
    return EXIT_SUCCESS;
}