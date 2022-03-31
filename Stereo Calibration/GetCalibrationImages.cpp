#include "pch.h"

bool FindChessboardCorners(cv::Mat &imgLeft, cv::Mat &imgRight)
{
    bool foundLeft = false, foundRight = false;
    std::vector<cv::Point2f> cornersLeft, cornersRight;

    // Find Chesscorners
    foundLeft = cv::findChessboardCorners(imgLeft, BOARD_SIZE, cornersLeft,
                                          cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);
    foundRight = cv::findChessboardCorners(imgRight, BOARD_SIZE, cornersRight,
                                           cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);

    if (!foundLeft || !foundRight)
        return false;

    cv::Mat grayLeft, grayRight, cornersImage;

    cv::cvtColor(imgLeft, grayLeft, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgRight, grayRight, cv::COLOR_BGR2GRAY);

    cv::cornerSubPix(grayLeft, cornersLeft, cv::Size(5, 5), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.1));
    cv::cornerSubPix(grayRight, cornersRight, cv::Size(5, 5), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.1));

    cv::drawChessboardCorners(grayLeft, BOARD_SIZE, cornersLeft, foundLeft);
    cv::drawChessboardCorners(grayRight, BOARD_SIZE, cornersRight, foundRight);

    cv::hconcat(grayLeft, grayRight, cornersImage);
    cv::imshow("Corners WebCam", cornersImage);
    cv::waitKey(0);

    return true;
}

int main()
{
    VideoCapture cam0("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
    VideoCapture cam1("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);

    if (!cam0.isOpened() || !cam1.isOpened())
    {
        std::cerr << "ERROR - NO CAMERAS DETECTED" << std::endl;
        return -1;
    }

    cout << "Resolution image cam0 : " << cv::Size(cam0.get(cv::CAP_PROP_FRAME_WIDTH), cam0.get(cv::CAP_PROP_FRAME_HEIGHT)) << endl;
    cout << "Frames per second using cam0 : " << cam0.get(cv::CAP_PROP_FPS) << endl;
    cout << "Resolution image cam1 : " << cv::Size(cam1.get(cv::CAP_PROP_FRAME_WIDTH), cam1.get(cv::CAP_PROP_FRAME_HEIGHT)) << endl;
    cout << "Frames per second using cam1 : " << cam1.get(cv::CAP_PROP_FPS) << endl;

    cout << "**** Instructions: **** " << endl;
    cout << "- Press (ESC) to exit!" << endl;

    int count = 1;
    Mat LeftImage, RightImage, StereoImage;
    while (cam0.read(LeftImage) && cam1.read(RightImage))
    {
        cv::hconcat(LeftImage, RightImage, StereoImage);
        cv::imshow("Stereo WebCam", StereoImage);

        if (FindChessboardCorners(LeftImage, RightImage))
        {
            cv::imwrite("../left/left" + to_string(count) + ".png", LeftImage);
            cv::imwrite("../right/right" + to_string(count++) + ".png", RightImage);
        }

        if ((char)waitKey(1) == 27)
            break;
    }

    return 0;
}
