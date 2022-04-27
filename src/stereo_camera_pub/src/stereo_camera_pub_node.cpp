#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include "std_msgs/String.h"
#include <sstream>

using namespace cv;
using namespace std;

constexpr char STEREO_PARAMS_PATH[] = "./Stereo Calibration/data/stereocalib.yml";

int main(int argc, char *argv[])
{
    VideoCapture cam0("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
    VideoCapture cam1("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);

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

    cout << "Resolution image cam0 : " << cv::Size(cam0.get(cv::CAP_PROP_FRAME_WIDTH), cam0.get(cv::CAP_PROP_FRAME_HEIGHT)) << endl;
    cout << "Frames per second using cam0 : " << cam0.get(cv::CAP_PROP_FPS) << endl;
    cout << "Resolution image cam1 : " << cv::Size(cam1.get(cv::CAP_PROP_FRAME_WIDTH), cam1.get(cv::CAP_PROP_FRAME_HEIGHT)) << endl;
    cout << "Frames per second using cam1 : " << cam1.get(cv::CAP_PROP_FPS) << endl;

    // Load Stereo Calibration Parameters

    cv::Mat Map1x, Map1y, Map2x, Map2y;
    cv::FileStorage stereodatafs;

    if (argv[1])
    {
        stereodatafs.open(STEREO_PARAMS_PATH, cv::FileStorage::READ);
        stereodatafs["Map1x"] >> Map1x;
        stereodatafs["Map1y"] >> Map1y;
        stereodatafs["Map2x"] >> Map2x;
        stereodatafs["Map2y"] >> Map2y;
        stereodatafs.release();

        cout << "Using instrinsic params: " << STEREO_PARAMS_PATH << endl;
    }

    // Init the Node Publisher and configure it.
    ros::init(argc, argv, "stereo_image_publisher");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub_left_camera = it.advertise("camera/left_image", 1);
    image_transport::Publisher pub_right_camera = it.advertise("camera/right_image", 1);

    sensor_msgs::ImagePtr imageLeftMsg;
    sensor_msgs::ImagePtr imageRightMsg;

    Mat cam0Frame;
    Mat cam1Frame;

    ros::Rate loop_rate(10);
    while (nh.ok())
    {
        if (!cam0.read(cam0Frame) ||
            !cam1.read(cam1Frame))
        {
            ROS_INFO("%s", "No frame info");
            return 1;
        }

        // Undistord stereo images
        if (argv[1])
        {
            remap(cam0Frame, cam0Frame, Map1x, Map1y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
            remap(cam1Frame, cam1Frame, Map2x, Map2y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
        }

        imageLeftMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cam0Frame).toImageMsg();
        imageRightMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cam1Frame).toImageMsg();

        pub_left_camera.publish(imageLeftMsg);
        pub_right_camera.publish(imageRightMsg);

        if (cv::waitKey(30) == 27)
            break;

        ros::spinOnce();
        loop_rate.sleep();
    }

    cam0.release();
    cam1.release();

    return 0;
}
