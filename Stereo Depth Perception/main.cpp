#include "pch.h"
#include "lib/plugin.h"

VOID DP_Process(
	CONST IN cv::Mat &LeftImage,
	CONST IN cv::Mat &RightImage,
	OUT cv::Mat &DepthImage)
{
	StereoDepthPerceptionLib::Compute(LeftImage, RightImage);
	StereoDepthPerceptionLib::GetDepthImage(DepthImage);
}

int main()
{
	std::string DATASET_PATH = "../res/";
	cv::VideoCapture cam0, cam1, cam2;
	cam0 = cv::VideoCapture(DATASET_PATH + "image_L/image (1).png", cv::CAP_IMAGES);
	cam1 = cv::VideoCapture(DATASET_PATH + "image_R/image (1).png", cv::CAP_IMAGES);
	cam2 = cv::VideoCapture(DATASET_PATH + "disparity/image (1).png", cv::CAP_IMAGES);

	std::cout << "Reading Dataset: " + DATASET_PATH << std::endl;
	if (!cam0.isOpened() || !cam1.isOpened())
	{
		std::cerr << "ERROR READING DATASET" << std::endl;
		return -1;
	}

	StereoDepthPerceptionLib::Setup(cv::Size(cam0.get(cv::CAP_PROP_FRAME_WIDTH), cam0.get(cv::CAP_PROP_FRAME_HEIGHT)));

	cv::Mat LeftImage, RightImage, DepthImage, DisparityImage;
	while (cam0.read(LeftImage) && cam1.read(RightImage) && cam2.read(DisparityImage))
	{
		DisparityImage.convertTo(DisparityImage, CV_8UC1);
		cv::applyColorMap(DisparityImage, DisparityImage, cv::COLORMAP_WINTER);

		auto TIME_START;
		DP_Process(LeftImage, RightImage, DepthImage);
		auto TIME_STOP;

		cv::Mat StereoImage;
		cv::hconcat(LeftImage, RightImage, StereoImage);
		cv::putText(StereoImage, "FPS: " + std::to_string(1 / TIME_RESULT), {10, 15}, 1, 1, CV_RGB(255, 0, 0));
		cv::imshow("Stereo WebCam", StereoImage);
		cv::imshow("Depth(CUDA)", DepthImage);
		cv::imshow("Disparity(Dataset)", DisparityImage);
		cv::waitKey(1);
	}

	return 0;
}