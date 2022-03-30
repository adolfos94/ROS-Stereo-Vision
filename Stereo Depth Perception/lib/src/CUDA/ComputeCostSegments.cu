#include "Wrapper.h"

__global__ void ComputeCostSegments_GPU(
	CONST IN int width,
	CONST IN int height,
	CONST IN int lenght,
	CONST IN float* pixelCost,
	CONST IN float* turboPixelImage,
	OUT float* costSegments)
{
	int idxX = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idxY = (blockIdx.y * blockDim.y) + threadIdx.y;
	int idxCUDA = idxX * width + idxY;
	int idxPixelCost = (idxY * height + idxX) * MaxDisparity;
	int idxTurboPixel;

	if (idxX >= 0 && idxY >= 0 && idxX < height && idxY < width)
	{
		idxTurboPixel = turboPixelImage[idxCUDA];

		for (int d = 0; d < MaxDisparity; ++d)
			atomicAdd(&costSegments[(idxTurboPixel * MaxDisparity) + d], pixelCost[idxPixelCost + d]);
		
	}
	__syncthreads();
}

VOID CUDA::ComputeCostSegments(
	CONST IN dim3& gridSize,
	CONST IN dim3& blockSize,
	CONST IN int& width,
	CONST IN int& height,
	CONST IN int& lenght,
	CONST IN float* pixelCost,
	CONST IN float* turboPixelImage,
	OUT float* costSegments)
{
	ComputeCostSegments_GPU << <gridSize, blockSize >> > (width, height, lenght, pixelCost, turboPixelImage, costSegments);
}