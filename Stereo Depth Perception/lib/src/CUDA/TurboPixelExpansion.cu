#include "Wrapper.h"

__global__ void TurboPixelExpansion_GPU(
	CONST IN int width,
	CONST IN int height,
	CONST IN int lenght,
	CONST IN float* grayImage,
	OUT float* turboPixelImage)
{
	int idxX = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idxY = (blockIdx.y * blockDim.y) + threadIdx.y;
	int idxCUDA = idxX * width + idxY;
	int idxk, idxCUDAk;
	int label;

	if (idxX >= 1 && idxY >= 1 && idxX < height - 1 && idxY < width - 1)
	{
		for (idxk = -1; idxk <= 1; ++idxk)
		{
			idxCUDAk = idxCUDA + (idxk * width) + idxk;

			if (abs(grayImage[idxCUDA] - grayImage[idxCUDAk]) <= 2)
				label = turboPixelImage[idxCUDAk];
		}
		turboPixelImage[idxCUDA] = label;
	}
}

VOID CUDA::TurboPixelExpansion(
	CONST IN dim3& gridSize,
	CONST IN dim3& blockSize,
	CONST IN int& width,
	CONST IN int& height,
	CONST IN int& lenght,
	CONST IN float* grayImage,
	OUT float* turboPixelImage)
{
	TurboPixelExpansion_GPU << < gridSize, blockSize >> > (width, height, lenght, grayImage, turboPixelImage);
}