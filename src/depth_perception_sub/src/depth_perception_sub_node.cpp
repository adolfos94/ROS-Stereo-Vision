#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "libStereoDepthPerceptionLib.h"

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> ApproximatePolicy;
typedef message_filters::Synchronizer<ApproximatePolicy> ApproximateSync;

void DisplayStereoWebCam(cv::Mat &imageLeft, cv::Mat &imageRight)
{
    cv::Mat stereoImage;
    cv::hconcat(imageLeft, imageRight, stereoImage);

    cv::imshow("Stereo WebCam", stereoImage);
}

void callback(const sensor_msgs::ImageConstPtr &l_image_msg,
              const sensor_msgs::ImageConstPtr &r_image_msg)
{
    cv::Mat LeftImage = cv_bridge::toCvShare(l_image_msg, "bgr8")->image;
    cv::Mat RightImage = cv_bridge::toCvShare(r_image_msg, "bgr8")->image;
    cv::Mat DepthMap;

    StereoDepthPerceptionLib::Compute(LeftImage, RightImage);
    StereoDepthPerceptionLib::GetDepthImage(DepthMap);

    DisplayStereoWebCam(LeftImage, RightImage);
    cv::imshow("DephBuffer", DepthMap);
    cv::waitKey(1);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "stereo_image_listener");
    ros::NodeHandle nh;

    image_transport::ImageTransport it(nh);
    image_transport::SubscriberFilter left_sub, right_sub;
    left_sub.subscribe(it, "camera/left_image", 1);
    right_sub.subscribe(it, "camera/right_image", 1);

    StereoDepthPerceptionLib::Setup(cv::Size(640, 480));

    boost::shared_ptr<ApproximateSync> approximate_sync;

    approximate_sync.reset(new ApproximateSync(ApproximatePolicy(1), left_sub, right_sub));
    approximate_sync->registerCallback(boost::bind(callback, _1, _2));

    ros::spin();
}