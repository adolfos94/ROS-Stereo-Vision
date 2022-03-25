#pragma once

#include "../../pch.h"

class WebCam
{
public:

	cv::Mat WebCamImage;
	cv::Mat DepthImage;
	cv::Mat LiDARImage;

	inline cv::Size GetWebCamResolution()
	{
		return resolution;
	}

	inline cv::Mat GetIntrinsicParameters()
	{
		return IntrinsicParameters;
	}

	VOID OnFrameAcquired(
		IN VOID(*process)(CONST IN cv::Mat&, CONST IN cv::Mat&, CONST IN cv::Mat&, OUT void*),
		OUT void* algorithm);

	VOID OnFrameAcquired(
		IN VOID(*process)(CONST IN cv::Mat&, CONST IN cv::Mat&, OUT void*),
		OUT void* algorithm);

	WebCam(CONST IN std::string& _datasetPath, bool stereo_cam = false);

	~WebCam() {};

private:
	int FPS;

	std::string datasetPath;

	cv::Size resolution;
	cv::Mat IntrinsicParameters;

	cv::Mat LeftImage;
	cv::Mat RightImage;

	cv::VideoCapture leftVideoCapture;
	cv::VideoCapture rightVideoCapture;

	cv::VideoCapture colorVideoCapture;
	cv::VideoCapture depthVideoCapture;

	bool LoadDepthBufferFromDataset(
		CONST IN std::string& path,
		OUT cv::Mat& depthBuffer);

	bool LoadLiDARBufferFromDataset(
		CONST IN std::string& path,
		OUT cv::Mat& LiDARImage);
};
