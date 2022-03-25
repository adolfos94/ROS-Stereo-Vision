#include "pch.h"
#include "source/WebCam/WebCam.h"
#include "StereoDepthPerceptionLib/pch.h"


VOID DP_Process(
	CONST IN cv::Mat& LeftImage,
	CONST IN cv::Mat& RightImage,
	OUT void* algorithm)
{
	cv::Mat depthMap;

	StereoDepthPerceptionLib::Compute(LeftImage, RightImage);
	StereoDepthPerceptionLib::GetDepthImage(depthMap);

#if VISUAL_DEBUG
	cv::imshow("DephBuffer", depthMap);
	cv::waitKey(1);
#endif
}

int main()
{
	WebCam webCam = WebCam(std::string("../resource/Dataset/"), true);

	
	StereoDepthPerceptionLib::Setup(webCam.GetWebCamResolution());
	
	webCam.OnFrameAcquired(DP_Process, nullptr);

	return 0;
}