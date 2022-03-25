#include "Wrapper.h"

__global__ void TurboPixelClustering_GPU(
	CONST IN int width,
	CONST IN int height,
	CONST IN int lenght,
	CONST IN float* grayImage,
	OUT float* turboPixelImage)
{
	int idxX = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idxY = (blockIdx.y * blockDim.y) + threadIdx.y;
	int idxCUDA = idxX * width + idxY;
	int idxXk, idxYk, idxCUDAk;

	int center = TurboPixelSize / 2;
	int label = INVALID_TURBO_PIXEL;
	float similarity, minSimilarity = 255.0f;

	if (idxX >= center && idxY >= center && idxX < height - center && idxY < width - center)
	{
		if (turboPixelImage[idxCUDA] == INVALID_TURBO_PIXEL)
		{
			for (idxXk = -center; idxXk <= center; ++idxXk)
			{
				for (idxYk = -center; idxYk <= center; ++idxYk)
				{
					idxCUDAk = idxCUDA + (idxXk * width) + idxYk;

					if (turboPixelImage[idxCUDAk] >= 0)
					{
						similarity = abs(grayImage[idxCUDA] - grayImage[idxCUDAk]);

						if (similarity <= minSimilarity)
						{
							minSimilarity = similarity;
							label = turboPixelImage[idxCUDAk];
						}
					}
				}
			}
			turboPixelImage[idxCUDA] = label;
		}
	}
}

VOID CUDA::TurboPixelClustering(
	CONST IN dim3& gridSize,
	CONST IN dim3& blockSize,
	CONST IN int& width,
	CONST IN int& height,
	CONST IN int& lenght,
	CONST IN float* grayImage,
	OUT float* turboPixelImage)
{
	TurboPixelClustering_GPU << <gridSize, blockSize >> > (width, height, lenght, grayImage, turboPixelImage);
}