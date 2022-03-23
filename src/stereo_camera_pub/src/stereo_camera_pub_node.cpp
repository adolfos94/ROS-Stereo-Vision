#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include "std_msgs/String.h"
#include <sstream>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    VideoCapture cam0("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
    VideoCapture cam1("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);

    if (!cam0.isOpened())
    {
        printf("cam0 is not opened.\n");
        return -1;
    }
    if (!cam1.isOpened())
    {
        printf("cam1 is not opened.\n");
        return -1;
    }

    // Init the Node Publisher and configure it.
    ros::init(argc, argv, "stereo_image_publisher");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub_left_camera = it.advertise("camera/left_image", 1);
    image_transport::Publisher pub_right_camera = it.advertise("camera/right_image", 1);

    sensor_msgs::ImagePtr imageLeftMsg;
    sensor_msgs::ImagePtr imageRightMsg;

    Mat leftFrame;
    Mat rightFrame;

    ros::Rate loop_rate(10);
    while (nh.ok())
    {
        if (!cam0.read(leftFrame) ||
            !cam1.read(rightFrame))
        {
            ROS_INFO("%s", "No frame info");
            return 1;
        }

        imageLeftMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", leftFrame).toImageMsg();
        imageRightMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", rightFrame).toImageMsg();

        pub_left_camera.publish(imageLeftMsg);
        pub_right_camera.publish(imageRightMsg);
        cv::waitKey(1);

        ros::spinOnce();
        loop_rate.sleep();
    }
}