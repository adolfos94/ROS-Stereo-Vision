#define IN
#define OUT
#define CONST const
#define VOID void
#define COUT std::cout
#define REMOVE "\r"
#define TAB "\t"
#define ENDL "\n"
#define STRINGIFY(VAR) (#VAR)
#define SIZE_PTR(T, M, N) (sizeof(T) * M * N)

#define VISUAL_DEBUG true

#define TIME_START start = std::chrono::high_resolution_clock::now()
#define TIME_STOP stop = std::chrono::high_resolution_clock::now()
#define TIME_RESULT (std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() / 1e9)

#include <iostream>
#include <fstream>
#include <random>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>

namespace StereoDepthPerceptionLib
{
	VOID Setup(CONST IN cv::Size size);

	VOID Compute(
		CONST IN cv::Mat &leftImage,
		CONST IN cv::Mat &rightImage);

	VOID GetDepthImage(OUT cv::Mat &depthImage);
}