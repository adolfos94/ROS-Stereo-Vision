#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>

// Defining the dimensions of checkerboard
constexpr int BOARD_WIDTH = 9;
constexpr int BOARD_HEIGHT = 7;
constexpr float SQUARE_SIZE = 20.00f; // mm

cv::Size m_board_size = cv::Size(BOARD_WIDTH, BOARD_HEIGHT);
std::vector<std::vector<cv::Point2f>> m_imagePointsLeft, m_imagePointsRight;
std::vector<std::vector<cv::Point3f>> m_ObjectPoints;
std::vector<cv::Point2f> m_cornersLeft, m_cornersRight;
std::vector<cv::Point3f> m_obj;

cv::Mat CM1 = cv::Mat(3, 3, CV_64FC1);
cv::Mat CM2 = cv::Mat(3, 3, CV_64FC1);

cv::Mat D1, D2;
cv::Mat R, T, E, F;
cv::Mat R1, R2, P1, P2, Q;
cv::Mat Map1x, Map1y, Map2x, Map2y;

std::vector<cv::Point3f> fillSquareDimensions()
{
    std::vector<cv::Point3f> obj;

    for (int i = 0; i < BOARD_HEIGHT; i++)
    {
        for (int j = 0; j < BOARD_WIDTH; j++)
        {
            obj.push_back(cv::Point3f((float)j * SQUARE_SIZE, (float)i * SQUARE_SIZE, 0));
        }
    }

    return obj;
}

void ExtractChessboardCorners(cv::Mat &imgLeft, cv::Mat &imgRight, int i)
{
    bool foundLeft = false, foundRight = false;
    cv::Mat grayLeft, grayRight;

    cv::cvtColor(imgLeft, grayLeft, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgRight, grayRight, cv::COLOR_BGR2GRAY);

    // Find Chesscorners
    foundLeft = cv::findChessboardCorners(imgLeft, m_board_size, m_cornersLeft,
                                          cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);
    foundRight = cv::findChessboardCorners(imgRight, m_board_size, m_cornersRight,
                                           cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);

    if (!foundLeft || !foundRight)
    {
        std::cerr << "Image #: " << i << "Chessboard find error!" << std::endl;
        return;
    }

    cv::cornerSubPix(grayLeft, m_cornersLeft, cv::Size(5, 5), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.1));
    cv::cornerSubPix(grayRight, m_cornersRight, cv::Size(5, 5), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.1));

    cv::drawChessboardCorners(imgLeft, m_board_size, m_cornersLeft, foundLeft);

    cv::drawChessboardCorners(imgRight, m_board_size, m_cornersRight, foundRight);

    m_imagePointsLeft.push_back(m_cornersLeft);
    m_imagePointsRight.push_back(m_cornersRight);
    m_ObjectPoints.push_back(m_obj);

    std::cout << "Image #: " << i << ". Found corners!" << std::endl;
}

void StereoCalibration(cv::Size imageSize)
{
    std::cout << "Starting Calibration..." << std::endl;
    cv::stereoCalibrate(m_ObjectPoints, m_imagePointsLeft, m_imagePointsRight,
                        CM1, D1, CM2, D2, imageSize, R, T, E, F,
                        cv::CALIB_SAME_FOCAL_LENGTH | cv::CALIB_ZERO_TANGENT_DIST,
                        cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 100, 1e-5));

    std::cout << "Calibration completed" << std::endl;
}

void StereoRectification(cv::Size imageSize)
{
    std::cout << "Starting Rectification..." << std::endl;
    stereoRectify(CM1, D1, CM2, D2, imageSize, R, T, R1, R2, P1, P2, Q);

    std::cout << "Rectification completed" << std::endl;
}

void StereoUndistortion(cv::Size imageSize)
{
    std::cout << "Applying Undistort..." << std::endl;

    initUndistortRectifyMap(CM1, D1, R1, P1, imageSize, CV_32FC1, Map1x, Map1y);
    initUndistortRectifyMap(CM2, D2, R2, P2, imageSize, CV_32FC1, Map2x, Map2y);

    std::cout << "Undistort completed" << std::endl;
}

void WriteCalibrationParams(std::string name)
{
    cv::FileStorage setereodatafs("../data/" + name, cv::FileStorage::WRITE);
    setereodatafs << "CM1" << CM1;
    setereodatafs << "CM2" << CM2;
    setereodatafs << "D1" << D1;
    setereodatafs << "D2" << D2;
    setereodatafs << "R" << R;
    setereodatafs << "T" << T;
    setereodatafs << "E" << E;
    setereodatafs << "F" << F;
    setereodatafs << "R1" << R1;
    setereodatafs << "R2" << R2;
    setereodatafs << "P1" << P1;
    setereodatafs << "P2" << P2;
    setereodatafs << "Q" << Q;
    setereodatafs << "Map1x" << Map1x;
    setereodatafs << "Map1y" << Map1y;
    setereodatafs << "Map2x" << Map2x;
    setereodatafs << "Map2y" << Map2y;
    setereodatafs.release();

    std::cout << "File wrote: " + name << std::endl;
}

int main()
{
    m_obj = fillSquareDimensions();

    // Path of the folder containing checkerboard images
    std::vector<cv::String> imagesPathLeft, imagesPathRight;
    cv::glob("../res/left_images/*.png", imagesPathLeft);
    cv::glob("../res/right_images/*.png", imagesPathRight);

    cv::Mat imgLeft, imgRight;
    for (int i = 0; i < imagesPathLeft.size(); ++i)
    {
        // Extract Images
        imgLeft = cv::imread(imagesPathLeft[i]);
        imgRight = cv::imread(imagesPathRight[i]);

        // Start Chessboard extraction..
        ExtractChessboardCorners(imgLeft, imgRight, i);
    }

    // Start Calibration..
    StereoCalibration(imgLeft.size());

    // Start Rectification..
    StereoRectification(imgLeft.size());

    // Start Undistortion..
    StereoUndistortion(imgLeft.size());

    // Write Stereo setereo calibration params
    WriteCalibrationParams("stereocalib.yml");

    std::cout << "Press any key to exit.." << std::endl;
    cv::waitKey(0);
    return 0;
}