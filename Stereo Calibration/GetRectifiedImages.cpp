#include "pch.h"

int main(int argc, char *argv[])
{
    VideoCapture cam0, cam1;

    if (argv[1]) // Use data
    {
        std::cout << "Using data images.." << std::endl;
        cam0 = VideoCapture("../res/left_images/01.png", cv::CAP_IMAGES);
        cam1 = VideoCapture("../res/right_images/01.png", cv::CAP_IMAGES);
    }
    else // Use webcam
    {
        cam0 = VideoCapture("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
        cam1 = VideoCapture("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
    }

    cout << "Resolution image cam0 : " << cv::Size(cam0.get(cv::CAP_PROP_FRAME_WIDTH), cam0.get(cv::CAP_PROP_FRAME_HEIGHT)) << endl;
    cout << "Frames per second using cam0 : " << cam0.get(cv::CAP_PROP_FPS) << endl;
    cout << "Resolution image cam1 : " << cv::Size(cam1.get(cv::CAP_PROP_FRAME_WIDTH), cam1.get(cv::CAP_PROP_FRAME_HEIGHT)) << endl;
    cout << "Frames per second using cam1 : " << cam1.get(cv::CAP_PROP_FPS) << endl;

    // Load Stereo Calibration Parameters
    cv::Mat Map1x, Map1y, Map2x, Map2y;
    cv::FileStorage setereodatafs = cv::FileStorage("../data/stereocalib.yml", cv::FileStorage::READ);
    setereodatafs["Map1x"] >> Map1x;
    setereodatafs["Map1y"] >> Map1y;
    setereodatafs["Map2x"] >> Map2x;
    setereodatafs["Map2y"] >> Map2y;
    setereodatafs.release();

    // Display Rectified Stereo Images..
    cv::Mat imageLeft, imageRight, imgLeftRect, imgRighRect;
    while (true)
    {
        cam0.read(imageLeft);
        cam1.read(imageRight);

        // Map1x, Map1y, Map2x, Map2y
        remap(imageLeft, imgLeftRect, Map1x, Map1y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
        remap(imageRight, imgRighRect, Map2x, Map2y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

        imshow("Left Rectified", imgLeftRect);
        imshow("Right Rectified", imgRighRect);

        if (cv::waitKey(30) == 27)
            break;
    }

    return 0;
}
