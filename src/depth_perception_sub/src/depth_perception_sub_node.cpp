#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include "libStereoDepthPerceptionLib.h"

cv::Mat LeftImage, RightImage;

void DisplayWebCam(std::string name, cv::Mat &imageMat)
{
    cv::imshow(name, imageMat);
    cv::waitKey(30);
}

void imageLeftCallback(const sensor_msgs::ImageConstPtr &msg)
{
    try
    {
        LeftImage = cv_bridge::toCvShare(msg, "bgr8")->image;
        DisplayWebCam("Left Image", LeftImage);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void imageRightCallback(const sensor_msgs::ImageConstPtr &msg)
{
    try
    {
        RightImage = cv_bridge::toCvShare(msg, "bgr8")->image;
        DisplayWebCam("Right Image", RightImage);

        cv::Mat depthMap;

        StereoDepthPerceptionLib::Compute(LeftImage, RightImage);
        StereoDepthPerceptionLib::GetDepthImage(depthMap);

        cv::imshow("DephBuffer", depthMap);
        cv::waitKey(1);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "stereo_image_listener");
    ros::NodeHandle nh;

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub_left_camera = it.subscribe("camera/left_image", 1, imageLeftCallback);
    image_transport::Subscriber sub_right_camera = it.subscribe("camera/right_image", 1, imageRightCallback);

    ros::spin();
}