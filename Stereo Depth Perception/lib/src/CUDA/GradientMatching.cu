#include "Wrapper.h"

__global__ void GradientMatching_GPU(
	CONST IN int width,
	CONST IN int height,
	CONST IN int lenght,
	CONST IN float* GxLeft,
	CONST IN float* GyLeft,
	CONST IN float* GxRight,
	CONST IN float* GyRight,
	OUT float* pixelCostLeft,
	OUT float* pixelCostRight)
{
	int idxX = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idxY = (blockIdx.y * blockDim.y) + threadIdx.y;
	int idxCUDA = idxX * width + idxY;
	int idxPixelCost = (idxY * height + idxX) * MaxDisparity;

	double diffGxLeft;
	double diffGyLeft;
	double diffGxRight;
	double diffGyRight;

	if (idxX >= 0 && idxY >= 0 && idxX < height && idxY < width)
	{
		for (int d = 0; d < MaxDisparity; ++d)
		{
			if (idxY - MaxDisparity >= 0)
			{
				diffGxLeft = abs(GxLeft[idxCUDA] - GxRight[idxCUDA - d]);
				pixelCostLeft[idxPixelCost + d] = diffGxLeft > 2 ? 2 : diffGxLeft;

				diffGyLeft = abs(GyLeft[idxCUDA] - GyRight[idxCUDA - d]);
				pixelCostLeft[idxPixelCost + d] += diffGyLeft > 2 ? 2 : diffGyLeft;
			}
			else
			{
				pixelCostLeft[idxPixelCost + d] = 4;
			}

			if (idxY + MaxDisparity < width)
			{
				diffGxRight = abs(GxLeft[idxCUDA + d] - GxRight[idxCUDA]);
				pixelCostRight[idxPixelCost + d] = diffGxRight > 2 ? 2 : diffGxRight;

				diffGyRight = abs(GyLeft[idxCUDA + d] - GyRight[idxCUDA]);
				pixelCostRight[idxPixelCost + d] += diffGyRight > 2 ? 2 : diffGyRight;
			}
			else
			{
				pixelCostRight[idxPixelCost + d] = 4;
			}
		}
	}
}

VOID CUDA::GradientMatching(
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
	OUT float* pixelCostRight)
{
	GradientMatching_GPU << <gridSize, blockSize >> > (width, height, lenght, GxLeft, GyLeft, GxRight, GyRight, pixelCostLeft, pixelCostRight);
}