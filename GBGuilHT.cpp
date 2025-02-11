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
    vector<Vec4f> positionBallard, positionGuil;

    // Set template's width and height.
    int w = templ.cols;
    int h = templ.rows;

    // Create ballard and set options.
    Ptr<GeneralizedHoughBallard> ballard = createGeneralizedHoughBallard();
    ballard->setMinDist(10); // Minimum distance between detected objects in pixels
    ballard->setLevels(360); // Set the number of rotation levels; 360 is maximal value.
    ballard->setDp(2); // Resolution of the accumulator used to detect centers of the objects
    ballard->setMaxBufferSize(1000); // Maximal size of inner buffers
    ballard->setVotesThreshold(40); // Accumulator threshold parameter (threshold for object detection)
    ballard->setCannyLowThresh(30); // The first threshold for the hysteresis procedure in Canny edge detector
    ballard->setCannyHighThresh(110); // The second threshold for the hysteresis procedure in Canny edge detector
    ballard->setTemplate(templ);

    // Create guil and set options.
    Ptr<GeneralizedHoughGuil> guil = createGeneralizedHoughGuil();
    guil->setMinDist(10); // Minimum distance between detected objects in pixels
    guil->setLevels(360); // Set the number of rotation levels; 360 is maximal value.
    guil->setDp(3); // Resolution of the accumulator used to detect centers of the objects
    guil->setMaxBufferSize(1000); // Maximal size of inner buffers
    guil->setMinAngle(0); // Minimal angle for the template in degrees
    guil->setMaxAngle(360); // Maximal angle for the template in degrees
    guil->setAngleStep(1); // Angle step in degrees
    guil->setAngleThresh(1500); // Angle threshold in degrees
    guil->setMinScale(0.5); // Minimal scale of the template
    guil->setMaxScale(2.0); // Maximal scale of the template
    guil->setScaleStep(0.05); // Scale step
    guil->setScaleThresh(50); // Scale threshold
    guil->setPosThresh(10); // Position threshold
    guil->setCannyLowThresh(30); // The first threshold for the hysteresis procedure in Canny edge detector
    guil->setCannyHighThresh(110); // The second threshold for the hysteresis procedure in Canny edge detector
    guil->setTemplate(templ);

    // Execute Ballard detection.
    ballard->detect(grayImage, positionBallard);

    // Execute Guil detection.
    guil->detect(grayImage, positionGuil);

    //  draw Ballard.
    // for (vector<Vec4f>::iterator iter = positionBallard.begin(); iter != positionBallard.end(); ++iter) {
    //     RotatedRect rRect = RotatedRect(Point2f((*iter)[0], (*iter)[1]), Size2f(w * (*iter)[2], h * (*iter)[2]), (*iter)[3]);
    //     Point2f vertices[4];
    //     rRect.points(vertices);
    //     for (int i = 0; i < 4; i++)
    //         line(image, vertices[i], vertices[(i + 1) % 4], Scalar(255, 0, 0), 6);
    // }

    // Draw Guil.
    // for (vector<Vec4f>::iterator iter = positionGuil.begin(); iter != positionGuil.end(); ++iter) {
    //     RotatedRect rRect = RotatedRect(Point2f((*iter)[0], (*iter)[1]), Size2f(w * (*iter)[2], h * (*iter)[2]), (*iter)[3]);
    //     Point2f vertices[4];
    //     rRect.points(vertices);
    //     for (int i = 0; i < 4; i++)
    //         line(image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);
    // }

    // imshow("result_img", image);
    // waitKey();
    return EXIT_SUCCESS;
}