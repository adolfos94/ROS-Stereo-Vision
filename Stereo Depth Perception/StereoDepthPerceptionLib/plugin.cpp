#include "src/Stereo Depth Perception.h"

namespace StereoDepthPerceptionLib
{
	VOID Setup(CONST IN cv::Size size)
	{
		StereoDepthPerception::Setup(size);
	}

	VOID Compute(
		CONST IN cv::Mat& leftImage,
		CONST IN cv::Mat& rightImage)
	{
		StereoDepthPerception::Compute(leftImage, rightImage);
	}

	VOID GetDepthImage(OUT cv::Mat& depthImage)
	{
		StereoDepthPerception::GetDepthImage(depthImage);
	}
}