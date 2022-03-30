#include "pch.h"
#include "lib/plugin.h"

cv::Ptr<cv::StereoBM> m_StereoBM;

VOID DP_Process(
	CONST IN cv::Mat &LeftImage,
	CONST IN cv::Mat &RightImage,
	OUT cv::Mat &DepthImage)
{
	StereoDepthPerceptionLib::Compute(LeftImage, RightImage);
	StereoDepthPerceptionLib::GetDepthImage(DepthImage);
}

VOID OCV_Process(
	CONST IN cv::Mat &LeftImage,
	CONST IN cv::Mat &RightImage,
	OUT cv::Mat &DisparityImage)
{
	cv::Mat GL, GR;
	cv::cvtColor(LeftImage, GL, cv::COLOR_BGR2GRAY);
	cv::cvtColor(RightImage, GR, cv::COLOR_BGR2GRAY);
	m_StereoBM->compute(GL, GR, DisparityImage);
	cv::normalize(DisparityImage, DisparityImage, 0, 255, cv::NORM_MINMAX, CV_8U);
}

int main()
{
	std::string DATASET_PATH = "../res/";
	cv::VideoCapture cam0, cam1, cam2;
	cam0 = cv::VideoCapture(DATASET_PATH + "image_L/image (1).png", cv::CAP_IMAGES);
	cam1 = cv::VideoCapture(DATASET_PATH + "image_R/image (1).png", cv::CAP_IMAGES);

	std::cout << "Reading Dataset: " + DATASET_PATH << std::endl;
	if (!cam0.isOpened() || !cam1.isOpened())
	{
		std::cerr << "ERROR READING DATASET" << std::endl;
		return -1;
	}

	// OpenCV
	m_StereoBM = cv::StereoBM::create();

	// CUDA
	StereoDepthPerceptionLib::Setup(cv::Size(cam0.get(cv::CAP_PROP_FRAME_WIDTH), cam0.get(cv::CAP_PROP_FRAME_HEIGHT)));

	cv::Mat LeftImage, RightImage, StereoImage, DepthImage, DisparityImage;
	while (cam0.read(LeftImage) && cam1.read(RightImage))
	{
		auto TIME_START;
		DP_Process(LeftImage, RightImage, DepthImage);
		auto TIME_STOP;

		OCV_Process(LeftImage, RightImage, DisparityImage);

		cv::hconcat(LeftImage, RightImage, StereoImage);
		cv::putText(StereoImage, "FPS: " + std::to_string(1 / TIME_RESULT), {10, 15}, 1, 1, CV_RGB(255, 0, 0));
		cv::imshow("Stereo WebCam", StereoImage);
		cv::imshow("Depth(CUDA)", DepthImage);
		cv::imshow("Disparity(Open CV)", DisparityImage);
		cv::waitKey(1);
	}

	return 0;
}