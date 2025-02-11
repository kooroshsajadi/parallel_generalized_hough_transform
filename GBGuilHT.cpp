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
    cvtColor(image, grayImage, COLOR_RGB2GRAY);

    // Create variable for location, scale and rotation of detected templates.
    vector<Vec4f> positionBallard, positionGuil;

    // Set template's width and height.
    int w = templ.cols;
    int h = templ.rows;

    // Create ballard and set options.
    Ptr<GeneralizedHoughBallard> ballard = createGeneralizedHoughBallard();
    ballard->setMinDist(10);
    ballard->setLevels(360);
    ballard->setDp(2);
    ballard->setMaxBufferSize(1000);
    ballard->setVotesThreshold(40);
    ballard->setCannyLowThresh(30);
    ballard->setCannyHighThresh(110);
    ballard->setTemplate(templ);

    // Create guil and set options.
    Ptr<GeneralizedHoughGuil> guil = createGeneralizedHoughGuil();
    guil->setMinDist(10);
    guil->setLevels(360);
    guil->setDp(3);
    guil->setMaxBufferSize(1000);
    guil->setMinAngle(0);
    guil->setMaxAngle(360);
    guil->setAngleStep(1);
    guil->setAngleThresh(1500);
    guil->setMinScale(0.5);
    guil->setMaxScale(2.0);
    guil->setScaleStep(0.05);
    guil->setScaleThresh(50);
    guil->setPosThresh(10);
    guil->setCannyLowThresh(30);
    guil->setCannyHighThresh(110);
    guil->setTemplate(templ);

    // Execute Ballard detection.
    ballard->detect(grayImage, positionBallard);

    // Execute Guil detection.
    guil->detect(grayImage, positionGuil);

    //  draw Ballard.
    for (vector<Vec4f>::iterator iter = positionBallard.begin(); iter != positionBallard.end(); ++iter) {
        RotatedRect rRect = RotatedRect(Point2f((*iter)[0], (*iter)[1]), Size2f(w * (*iter)[2], h * (*iter)[2]), (*iter)[3]);
        Point2f vertices[4];
        rRect.points(vertices);
        for (int i = 0; i < 4; i++)
            line(image, vertices[i], vertices[(i + 1) % 4], Scalar(255, 0, 0), 6);
    }

    // Draw Guil.
    for (vector<Vec4f>::iterator iter = positionGuil.begin(); iter != positionGuil.end(); ++iter) {
        RotatedRect rRect = RotatedRect(Point2f((*iter)[0], (*iter)[1]), Size2f(w * (*iter)[2], h * (*iter)[2]), (*iter)[3]);
        Point2f vertices[4];
        rRect.points(vertices);
        for (int i = 0; i < 4; i++)
            line(image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);
    }

    imshow("result_img", image);
    waitKey();
    return EXIT_SUCCESS;
}