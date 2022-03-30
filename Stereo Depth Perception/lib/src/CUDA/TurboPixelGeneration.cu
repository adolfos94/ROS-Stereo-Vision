#include "Wrapper.h"

__global__ void TurboPixelGeneration_GPU(
	CONST IN int width,
	CONST IN int height,
	CONST IN int lenght,
	CONST IN float* grayImage,
	OUT float* turboPixelImage)
{
	int idxX = ((blockIdx.x * blockDim.x) + threadIdx.x) * TurboPixelSize;
	int idxY = ((blockIdx.y * blockDim.y) + threadIdx.y) * TurboPixelSize;
	int idxCUDA = idxX * width + idxY;
	int idxXk, idxYk, idxPivot, idxPivotk, idxGrid;

	int center = TurboPixelSize / 2;

	if (idxX >= 0 && idxY >= 0 && idxX < height && idxY < width)
	{
		idxPivot = idxCUDA + center + (center * width);
		idxGrid = (idxX * (width / TurboPixelSize) + idxY) / TurboPixelSize;

		for (idxXk = -center; idxXk <= center; ++idxXk)
		{
			for (idxYk = -center; idxYk <= center; ++idxYk)
			{
				idxPivotk = idxPivot + (idxXk * width) + idxYk;

				if (abs(grayImage[idxPivot] - grayImage[idxPivotk]) <= 1 && idxX > center && idxY > center)
					turboPixelImage[idxPivotk] = idxGrid;
				else if (idxX < height - TurboPixelSize && idxY < width - TurboPixelSize && idxX > center && idxY > center)
					turboPixelImage[idxPivotk] = INVALID_TURBO_PIXEL;
				else
					turboPixelImage[idxPivotk] = idxGrid;
			}
		}
	}
}

VOID CUDA::TurboPixelGeneration(
	CONST IN dim3& gridSize,
	CONST IN dim3& blockSize,
	CONST IN int& width,
	CONST IN int& height,
	CONST IN int& lenght,
	CONST IN float* grayImage,
	OUT float* turboPixelImage)
{
	TurboPixelGeneration_GPU << <gridSize, blockSize >> > (width, height, lenght, grayImage, turboPixelImage);
}