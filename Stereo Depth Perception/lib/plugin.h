#pragma once

#include "../pch.h"

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