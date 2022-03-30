#include "Wrapper.h"

__constant__ float tau = 0.2f;
__constant__ float sigma = 10.00f;

__global__ void GraphConstruction_GPU(
	CONST IN int width,
	CONST IN int height,
	CONST IN int lenght,
	CONST IN float* grayImage,
	CONST IN float* turboPixelImage,
	OUT float* graph,
	OUT float* intensities)
{
	int idxX = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idxY = (blockIdx.y * blockDim.y) + threadIdx.y;
	int idxCUDA = idxX * width + idxY;

	int idxTurboPixel, idxNeighborTurboPixel;
	int neighboors[4] = { -1, 1, -width, width };

	if (idxX >= 0 && idxY >= 0 && idxX < height && idxY < width)
	{
		idxTurboPixel = turboPixelImage[idxCUDA];
		atomicAdd(&intensities[(idxTurboPixel * 3) + 0], grayImage[idxCUDA]);
		atomicAdd(&intensities[(idxTurboPixel * 3) + 1], 1);
	}

	__syncthreads();

	float diff = 0;
	if (idxX > 0 && idxY > 0 && idxX < height - 1 && idxY < width - 1)
	{
		for (int i = 0; i < 4; ++i)
		{
			idxNeighborTurboPixel = turboPixelImage[idxCUDA + neighboors[i]];

			if (idxTurboPixel != idxNeighborTurboPixel)
			{
				diff = abs(
					(intensities[idxTurboPixel] / intensities[(idxTurboPixel * 3) + 1]) -
					(intensities[idxNeighborTurboPixel] / intensities[(idxNeighborTurboPixel * 3) + 1]));

				graph[idxTurboPixel + (lenght * idxNeighborTurboPixel)] = diff < 255  
					? (1 - tau) * exp(-(diff * diff) / sigma) + tau 
					: 0.0001;
			}
		}
	}
}

VOID CUDA::GraphConstruction(
	CONST IN dim3& gridSize,
	CONST IN dim3& blockSize,
	CONST IN int& width,
	CONST IN int& height,
	CONST IN int& lenght,
	CONST IN float* grayImage,
	CONST IN float* turboPixelImage,
	OUT float* graph,
	OUT float* intensities)
{
	GraphConstruction_GPU << <gridSize, blockSize >> > (width, height, lenght, grayImage, turboPixelImage, graph, intensities);
}