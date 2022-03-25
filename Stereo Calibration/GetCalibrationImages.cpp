#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
    VideoCapture cam0("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
    VideoCapture cam1("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);

    cout << "Resolution image cam0 : " << cv::Size(cam0.get(cv::CAP_PROP_FRAME_WIDTH), cam0.get(cv::CAP_PROP_FRAME_HEIGHT)) << endl;
    cout << "Frames per second using cam0 : " << cam0.get(cv::CAP_PROP_FPS) << endl;
    cout << "Resolution image cam1 : " << cv::Size(cam1.get(cv::CAP_PROP_FRAME_WIDTH), cam1.get(cv::CAP_PROP_FRAME_HEIGHT)) << endl;
    cout << "Frames per second using cam1 : " << cam1.get(cv::CAP_PROP_FPS) << endl;

    int count;
    count = 1;

    while (1)
    {
        Mat frame0;
        Mat frame1;

        cam0 >> frame0;
        cam1 >> frame1;

        //-- 1. Read the images

        Mat imgLeft;
        Mat imgRight;

        cvtColor(frame0, imgRight, COLOR_RGB2GRAY);
        cvtColor(frame1, imgLeft, COLOR_RGB2GRAY);

        //-- 2. Display the images

        namedWindow("imgLeft", WINDOW_AUTOSIZE);
        namedWindow("imgRight", WINDOW_AUTOSIZE);

        imshow("imgLeft", imgLeft);
        imshow("imgRight", imgRight);

        cv::imwrite("../left/left" + to_string(count) + ".png", frame1);
        cv::imwrite("../right/right" + to_string(count) + ".png", frame0);

        if ((char)waitKey(1500) == 27)
        {
            break;
        }

        count++;
    }

    return 0;
}
