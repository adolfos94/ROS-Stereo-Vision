#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Defining the dimensions of checkerboard
constexpr int BOARD_WIDTH = 9;
constexpr int BOARD_HEIGHT = 7;
constexpr float SQUARE_SIZE = 20.00f; // mm

cv::Size BOARD_SIZE = cv::Size(BOARD_WIDTH, BOARD_HEIGHT);