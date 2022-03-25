#include "WebCam.h"

bool  WebCam::LoadDepthBufferFromDataset(
	CONST IN std::string& path,
	OUT cv::Mat& depthBuffer)
{
	depthBuffer = cv::Mat(WebCamImage.size(), CV_32F);

	std::ifstream file(datasetPath + "depth/" + path + ".txt");
	std::string line;
	int rows = 0;
	while (std::getline(file, line))
	{
		std::stringstream ssline(line);
		std::string val;
		int cols = 0;
		while (std::getline(ssline, val, ','))
		{
			depthBuffer.at<float>(rows, cols) = stof(val);
			cols++;
		}
		rows++;
	}

	if (rows == 0)
	{
		std::cerr << "Failed to load Depth Dataset" << std::endl;
		return false;
	}

	return true;
}

bool WebCam::LoadLiDARBufferFromDataset(
	CONST IN std::string& path,
	OUT cv::Mat& LiDARImage)
{
	LiDARImage = cv::Mat();

	std::ifstream file(datasetPath + "lidar/" + path + ".txt");
	std::string line;
	int rows = 0;
	while (std::getline(file, line))
	{
		std::stringstream ssline(line);
		std::string val;
		while (std::getline(ssline, val, ','))
		{
			LiDARImage.push_back(stof(val));
		}
		rows++;
	}

	if (rows == 0)
	{
		std::cerr << "Failed to load LiDAR Dataset" << std::endl;
		return false;
	}

	LiDARImage = LiDARImage.reshape(1, rows);

	return true;
}

VOID WebCam::OnFrameAcquired(
	IN VOID(*process)(CONST IN cv::Mat&, CONST IN cv::Mat&, CONST IN cv::Mat&, OUT void*),
	OUT void* algorithm)
{
	int i = 0;

	while (// Extraction.
		colorVideoCapture.read(WebCamImage) &&
		LoadDepthBufferFromDataset(std::to_string(++i), DepthImage) &&
		LoadLiDARBufferFromDataset(std::to_string(i), LiDARImage))
	{
		auto TIME_START;
		// Processing.
		WebCamImage.convertTo(WebCamImage, CV_32F);
		process(WebCamImage, DepthImage, LiDARImage, algorithm);
		auto TIME_STOP;

#if VISUAL_DEBUG
		WebCamImage.convertTo(WebCamImage, CV_8UC1);
		cv::putText(WebCamImage, "FPS: " + std::to_string(1 / TIME_RESULT), { 10,15 }, 1, 1, CV_RGB(255, 0, 0));

		cv::imshow("WebCam", WebCamImage);
		cv::waitKey(1);
#else
		COUT << REMOVE << "FPS: " << std::to_string(1 / TIME_RESULT);

#endif // VISUAL_DEBUG
	}
}

VOID WebCam::OnFrameAcquired(
	IN VOID(*process)(CONST IN cv::Mat&, CONST IN cv::Mat&, OUT void*),
	OUT void* algorithm)
{
	while (// Extraction.
		leftVideoCapture.read(LeftImage) && rightVideoCapture.read(RightImage))
	{
		auto TIME_START;
		// Processing.
		LeftImage.convertTo(LeftImage, CV_32F);
		RightImage.convertTo(RightImage, CV_32F);

		process(LeftImage, RightImage, algorithm);
		auto TIME_STOP;

#if VISUAL_DEBUG
		LeftImage.convertTo(LeftImage, CV_8UC1);
		cv::putText(LeftImage, "FPS: " + std::to_string(1 / TIME_RESULT), { 10,15 }, 1, 1, CV_RGB(255, 0, 0));

		RightImage.convertTo(RightImage, CV_8UC1);
		cv::putText(RightImage, "FPS: " + std::to_string(1 / TIME_RESULT), { 10,15 }, 1, 1, CV_RGB(255, 0, 0));

		cv::imshow("Left WebCam", LeftImage);
		cv::imshow("Right WebCam", RightImage);
		cv::waitKey(0);
#else
		COUT << REMOVE << "FPS: " << std::to_string(1 / TIME_RESULT);

#endif // VISUAL_DEBUG
	}
}

WebCam::WebCam(CONST IN std::string& _datasetPath, bool stereo_cam)
{
	datasetPath = _datasetPath;

	if (!stereo_cam)
	{
		colorVideoCapture = cv::VideoCapture(datasetPath + "rgb/rgb000001.png");
		if (!colorVideoCapture.isOpened())
			COUT << "Error opening color dataset" << ENDL;

		FPS = colorVideoCapture.get(cv::CAP_PROP_FPS);

		resolution.width = colorVideoCapture.get(cv::CAP_PROP_FRAME_WIDTH);
		resolution.height = colorVideoCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
	}
	else
	{
		leftVideoCapture = cv::VideoCapture(datasetPath + "stereo/Left/rgb000001.png");
		rightVideoCapture = cv::VideoCapture(datasetPath + "stereo/Right/rgb000001.png");

		if (!leftVideoCapture.isOpened() || !rightVideoCapture.isOpened())
			COUT << "Error opening stereo dataset" << ENDL;

		FPS = leftVideoCapture.get(cv::CAP_PROP_FPS);

		resolution.width = leftVideoCapture.get(cv::CAP_PROP_FRAME_WIDTH);
		resolution.height = leftVideoCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
	}

	IntrinsicParameters = cv::Mat::zeros(3, 3, CV_32F);
	IntrinsicParameters.at<float>(0, 0) = 582.10184f; //Fx
	IntrinsicParameters.at<float>(1, 1) = 582.10184f; //Fx
	IntrinsicParameters.at<float>(0, 2) = 336.50000f; //Cx
	IntrinsicParameters.at<float>(1, 2) = 188.50000f; //Cy

	COUT << "WebCam Capturing FPS: " << FPS << ENDL;
	COUT << "WebCam Resolution: " << resolution << ENDL;
}