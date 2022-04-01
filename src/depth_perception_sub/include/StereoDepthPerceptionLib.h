#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

namespace StereoDepthPerceptionLib
{
	void Setup(const cv::Size size);

	void Compute(
		const cv::Mat &leftImage,
		const cv::Mat &rightImage);

	void GetDepthImage(cv::Mat &depthImage);
}