#include "Wrapper.h"

__global__ void ComputeCostCoordinates_GPU(
	CONST IN int width,
	CONST IN int height,
	CONST IN int lenght,
	CONST IN float* grayImage,
	CONST IN float* turboPixelImage,
	OUT float* costCoordinates)
{
	int idxX = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idxY = (blockIdx.y * blockDim.y) + threadIdx.y;
	int idxCUDA = idxX * width + idxY;
	int idxTurboPixel;

	if (idxX >= 0 && idxY >= 0 && idxX < height && idxY < width)
	{
		idxTurboPixel = turboPixelImage[idxCUDA];

		atomicAdd(&costCoordinates[(idxTurboPixel * CostParams) + 0], idxY);
		atomicAdd(&costCoordinates[(idxTurboPixel * CostParams) + 1], idxX);
		atomicAdd(&costCoordinates[(idxTurboPixel * CostParams) + 2], grayImage[idxCUDA]);
		atomicAdd(&costCoordinates[(idxTurboPixel * CostParams) + 3], 1);
	}

	__syncthreads();
}

VOID CUDA::ComputeCostCoordinates(
	CONST IN dim3& gridSize,
	CONST IN dim3& blockSize,
	CONST IN int& width,
	CONST IN int& height,
	CONST IN int& lenght,
	CONST IN float* grayImage,
	CONST IN float* turboPixelImage,
	OUT float* costCoordinates)
{
	ComputeCostCoordinates_GPU << <gridSize, blockSize >> > (width, height, lenght, grayImage, turboPixelImage, costCoordinates);
}