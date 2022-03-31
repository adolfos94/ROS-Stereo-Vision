#pragma once

#include "../plugin.h"

class StereoDepthPerception
{
public:
	static VOID Setup(CONST IN cv::Size size);

	static VOID Compute(
		CONST IN cv::Mat &leftImage,
		CONST IN cv::Mat &rightImage);

	static VOID GetDepthImage(OUT cv::Mat &depthImage);

private:
	// TODO: Free Images at the end of the Processing.
	static float *PenaltyMatrix_GPU;

	static float *LeftImage_GPU;
	static float *RightImage_GPU;

	static float *GrayLeftImage_GPU;
	static float *GrayRightImage_GPU;

	static float *GxLeft_GPU;
	static float *GyLeft_GPU;
	static float *GxRight_GPU;
	static float *GyRight_GPU;

	static float *PixelCostLeft_GPU;
	static float *PixelCostRight_GPU;

	static float *TurboPixelLeft_GPU;
	static float *TurboPixelRight_GPU;

	static float *CostCoordinatesLeft_GPU;
	static float *CostCoordinatesRight_GPU;

	static float *CostSegmentsLeft_GPU;
	static float *CostSegmentsRight_GPU;

	static float *GraphLeft_GPU;
	static float *GraphRight_GPU;
	static float *IntensitiesLeft_GPU;
	static float *IntensitiesRight_GPU;
	static float *NeighborhoodsLeft_GPU;
	static float *NeighborhoodsRight_GPU;

	static float *IterativeCostLeft_GPU;
	static float *IterativeCostRight_GPU;
	static float *InitialCostLeft_GPU;
	static float *InitialCostRight_GPU;

	static float *DisparityLeft_GPU;
	static float *DisparityRight_GPU;
	static float *Disparity_GPU;
	static float *Depth_GPU;

	static int Width;
	static int Height;
	static int Lenght;
	static int NumberGridSegments;
	static size_t DataLenght;

	static int MaxThreadsPerBlock;
	static dim3 BlockSize, blockSize;
	static dim3 GridSize, gridSize;

	static VOID CUDA2MAT(
		CONST IN float *image_GPU,
		CONST IN int &width,
		CONST IN int &height,
		OUT cv::Mat &image,
		bool disp = false);

	static VOID MAT2CUDA(
		CONST IN cv::Mat &image,
		OUT float **image_GPU);

	static VOID SetMemoryAllocations();

	static VOID SetPenaltyMatrix();

	static VOID RestartProbabilities();
	static VOID ComputeVisibility();
	static VOID ComputeFidelity();

	static VOID CropImages(
		CONST IN cv::Mat &leftImage,
		CONST IN cv::Mat &rightImage);

	static VOID ConvertRGB2GrayScale(CONST IN int channels);

	static VOID GradientOperations();

	static VOID GradientMatching();

	static VOID TurboPixelGeneration();

	static VOID TurboPixelClustering();

	static VOID TurboPixelExpansion(CONST IN int iterations = 5);

	static VOID ComputeCostCoordinates();

	static VOID ComputeCostSegments();

	static VOID ComputeCostNormalization();

	static VOID GraphConstruction();

	static VOID NeighborhoodsConstruction();

	static VOID RandomWalkWithRestart(CONST IN int iterations = 5);

	static VOID GetDisparityMap();

	static VOID GetDepthMap();
};
