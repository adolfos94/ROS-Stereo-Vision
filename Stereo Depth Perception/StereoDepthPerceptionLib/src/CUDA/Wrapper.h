#include "../../pch.h"

#define INVALID_TURBO_PIXEL -1

__constant__  int MaxDisparity = 64;
__constant__ int TurboPixelSize = 9;
__constant__ int CostParams = 4;
__constant__ int Neighborhoods = 100;
__constant__ float TAU = 7.0f;
__constant__ float PSI = 85.0f;
__constant__ float RestartEpsilon = 0.0015f;

namespace CUDA
{
	VOID GetDeviceInfo(OUT int& maxThreadsPerBlock);

	VOID ConvertRGB2GrayScale(
		CONST IN dim3& gridSize,
		CONST IN dim3& blockSize,
		CONST IN int& width,
		CONST IN int& height,
		CONST IN int& lenght,
		CONST IN int& channels,
		CONST IN float* imageRGB,
		OUT float* grayImage);

	VOID GradientConvolution(
		CONST IN dim3& gridSize,
		CONST IN dim3& blockSize,
		CONST IN int& width,
		CONST IN int& height,
		CONST IN int& lenght,
		CONST IN float* grayImage,
		OUT float* GxImage,
		OUT float* GyImage);

	VOID GradientMatching(
		CONST IN dim3& gridSize,
		CONST IN dim3& blockSize,
		CONST IN int& width,
		CONST IN int& height,
		CONST IN int& lenght,
		CONST IN float* GxLeft,
		CONST IN float* GyLeft,
		CONST IN float* GxRight,
		CONST IN float* GyRight,
		OUT float* pixelCostLeft,
		OUT float* pixelCostRight);

	VOID TurboPixelGeneration(
		CONST IN dim3& gridSize,
		CONST IN dim3& blockSize,
		CONST IN int& width,
		CONST IN int& height,
		CONST IN int& lenght,
		CONST IN float* grayImage,
		OUT float* turboPixelImage);

	VOID TurboPixelClustering(
		CONST IN dim3& gridSize,
		CONST IN dim3& blockSize,
		CONST IN int& width,
		CONST IN int& height,
		CONST IN int& lenght,
		CONST IN float* grayImage,
		OUT float* turboPixelImage);

	VOID TurboPixelExpansion(
		CONST IN dim3& gridSize,
		CONST IN dim3& blockSize,
		CONST IN int& width,
		CONST IN int& height,
		CONST IN int& lenght,
		CONST IN float* grayImage,
		OUT float* turboPixelImage);

	VOID ComputeCostCoordinates(
		CONST IN dim3& gridSize,
		CONST IN dim3& blockSize,
		CONST IN int& width,
		CONST IN int& height,
		CONST IN int& lenght,
		CONST IN float* grayImage,
		CONST IN float* turboPixelImage,
		OUT float* costCoordinates);

	VOID ComputeCostSegments(
		CONST IN dim3& gridSize,
		CONST IN dim3& blockSize,
		CONST IN int& width,
		CONST IN int& height,
		CONST IN int& lenght,
		CONST IN float* pixelCost,
		CONST IN float* turboPixelImage,
		OUT float* costSegments);

	VOID ComputeCostNormalization(
		CONST IN dim3& gridSize,
		CONST IN dim3& blockSize,
		CONST IN int& lenght,
		OUT float* costCoordinates,
		OUT float* costSegments);

	VOID GraphConstruction(
		CONST IN dim3& gridSize,
		CONST IN dim3& blockSize,
		CONST IN int& width,
		CONST IN int& height,
		CONST IN int& lenght,
		CONST IN float* grayImage,
		CONST IN float* turboPixelImage,
		OUT float* graph,
		OUT float* intensities);

	VOID NeighborhoodsConstruction(
		CONST IN dim3& gridSize,
		CONST IN dim3& blockSize,
		CONST IN int& lenght,
		OUT float* graph,
		OUT float* intensities,
		OUT float* neighborhoods);

	VOID ComputeVisibility(
		CONST IN dim3& gridSize,
		CONST IN dim3& blockSize,
		CONST IN int& width,
		CONST IN int& height,
		CONST IN int& lenght,
		CONST IN float* costCoordinatesLeft,
		CONST IN float* costCoordinatesRight,
		CONST IN float* turboPixelImageLeft,
		CONST IN float* turboPixelImageRight,
		OUT float* costSegmentsLeft,
		OUT float* costSegmentsRight,
		OUT float* disparityLeft,
		OUT float* disparityRight);

	VOID ComputeFidelity(
		CONST IN dim3& gridSize,
		CONST IN dim3& blockSize,
		CONST IN int& lenght,
		CONST IN float* penaltyMatrix,
		CONST IN float* neighborhoodsLeft,
		CONST IN float* neighborhoodsRight,
		OUT float* costSegmentsLeft,
		OUT float* costSegmentsRight,
		OUT float* disparityLeft,
		OUT float* disparityRight);

	VOID GetDisparityMap(
		CONST IN dim3& gridSize,
		CONST IN dim3& blockSize,
		CONST IN int& lenght,
		CONST IN float* costSegments,
		OUT float* disparity);

	VOID GetDepthMap(
		CONST IN dim3& gridSize,
		CONST IN dim3& blockSize,
		CONST IN int& width,
		CONST IN int& height,
		CONST IN int& lenght,
		CONST IN float* costSegments,
		CONST IN float* pixelCost,
		CONST IN float* turboPixelImage,
		OUT float* depth);
};

namespace CUBLAS {
	// Multiply the matrix A and B on GPU and save the result in C
	//C(m, n) = A(m, k) * B(k, n)
	VOID MatrixMultiplication(
		CONST IN float* A,
		CONST IN float* B,
		CONST IN int m,
		CONST IN int k,
		CONST IN int n,
		OUT float* C);

	// Addition the matrix A and B
	// B(m,n) = A(m,n) + B(m,n)
	VOID MatrixAddition(
		CONST IN float* A,
		CONST IN int m,
		CONST IN int n,
		OUT float* B);

	// Multiply the matrix A by an alpha
	// A(m,n) = alpha * A(m,n)
	VOID ScalarMatrixMultiplication(
		CONST IN float alpha,
		CONST IN int m,
		CONST IN int n,
		OUT float* A);
};