#include "Wrapper.h"

__global__ void ConvertRGB2GrayScale_GPU(
	CONST IN int width,
	CONST IN int height,
	CONST IN int lenght,
	CONST IN int channels,
	CONST IN float* imageRGB,
	OUT float* grayImage)
{
	int idxX = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idxY = (blockIdx.y * blockDim.y) + threadIdx.y;
	int idxCUDA = idxX * width + idxY;

	if (idxX >= 0 && idxY >= 0 && idxX < height && idxY < width)
	{
		grayImage[idxCUDA] =
			0.2126f * imageRGB[idxCUDA * channels + 2] +
			0.7152f * imageRGB[idxCUDA * channels + 1] +
			0.0722f * imageRGB[idxCUDA * channels + 0];
	}
}

VOID CUDA::ConvertRGB2GrayScale(
	CONST IN dim3& gridSize,
	CONST IN dim3& blockSize,
	CONST IN int& width,
	CONST IN int& height,
	CONST IN int& lenght,
	CONST IN int& channels,
	CONST IN float* imageRGB,
	OUT float* grayImage)
{
	ConvertRGB2GrayScale_GPU << <gridSize, blockSize >> > (width, height, lenght, channels, imageRGB, grayImage);
}