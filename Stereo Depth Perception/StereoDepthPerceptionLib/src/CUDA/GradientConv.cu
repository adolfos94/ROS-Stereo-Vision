#include "Wrapper.h"

__constant__  float Gx_Kernel[25] = {
	0.0104f, 0.0208f, 0.0f, -0.0208f, -0.0104f,
	0.0417f, 0.0833f, 0.0f, -0.0833f, -0.0417f,
	0.0625f, 0.1250f, 0.0f, -0.1250f, -0.0625f,
	0.0417f, 0.0833f, 0.0f, -0.0833f, -0.0417f,
	0.0104f, 0.0208f, 0.0f, -0.0208f, -0.0104f
};
__constant__  float Gy_Kernel[25] = {
	0.01040f, 0.04170f, 0.06250f, 0.04170f, 0.01040f,
	0.02080f, 0.08330f, 0.12500f, 0.08330f, 0.02080f,
	0.00000f, 0.00000f, 0.00000f, 0.00000f, 0.00000f,
	-0.0208f, -0.0833f, -0.1250f, -0.0833f, -0.0208f,
	-0.0104f, -0.0417f, -0.0625f, -0.0417f, -0.0104f
};
__global__ void GradientConvolution_GPU(
	CONST IN int width,
	CONST IN int height,
	CONST IN int lenght,
	CONST IN float* grayImage,
	OUT float* GxImage,
	OUT float* GyImage)
{
	int idxX = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idxY = (blockIdx.y * blockDim.y) + threadIdx.y;
	int idxCUDA = idxX * width + idxY;
	int idxXk, idxYk, idxCUDAk;
	int idxKernel = 0;

	float convGx = 0.0f;
	float convGy = 0.0f;

	if (idxX >= 2 && idxY >= 2 && idxX < height - 2 && idxY < width - 2)
	{
		for (idxXk = -2; idxXk <= 2; ++idxXk)
		{
			for (idxYk = -2; idxYk <= 2; ++idxYk)
			{
				idxCUDAk = idxCUDA + (idxXk * width) + idxYk;

				convGx += grayImage[idxCUDAk] * Gx_Kernel[idxKernel];
				convGy += grayImage[idxCUDAk] * Gy_Kernel[idxKernel];

				idxKernel++;
			}
		}
		GxImage[idxCUDA] = convGx;
		GyImage[idxCUDA] = convGy;
	}
}

VOID CUDA::GradientConvolution(
	CONST IN dim3& gridSize,
	CONST IN dim3& blockSize,
	CONST IN int& width,
	CONST IN int& height,
	CONST IN int& lenght,
	CONST IN float* grayImage,
	OUT float* GxImage,
	OUT float* GyImage)
{
	GradientConvolution_GPU << <gridSize, blockSize >> > (width, height, lenght, grayImage, GxImage, GyImage);
}