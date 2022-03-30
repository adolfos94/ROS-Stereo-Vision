#include "Wrapper.h"

__global__ void GetDepthMap_GPU(
	CONST IN int width,
	CONST IN int height,
	CONST IN int lenght,
	CONST IN float* costSegments,
	CONST IN float* pixelCost,
	CONST IN float* turboPixelImage,
	OUT float* depth)
{
	int idxX = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idxY = (blockIdx.y * blockDim.y) + threadIdx.y;
	int idxCUDA = idxX * width + idxY;

	if (idxX >= 0 && idxY >= 0 && idxX < height && idxY < width)
	{
		int segmentIDX = 0;
		float minCost = FLT_MAX;
		float disparityCost;

		segmentIDX = turboPixelImage[idxCUDA];

		if (segmentIDX > 0)
		{
			if (costSegments[segmentIDX] > 0)
			{
				for (int d = 0; d < MaxDisparity; ++d)
				{
					disparityCost =
						(0.0005f * pixelCost[((idxY * height + idxX) * MaxDisparity) + d]) +
						(0.9995f * costSegments[(segmentIDX * MaxDisparity) + d]);

					if (disparityCost < minCost)
					{
						minCost = disparityCost;
						depth[idxCUDA] = d;
					}
				}
			}
			else
				depth[idxCUDA] = 0;
		}
		else
			depth[idxCUDA] = 0;
	}
}

VOID CUDA::GetDepthMap(
	CONST IN dim3& gridSize,
	CONST IN dim3& blockSize,
	CONST IN int& width,
	CONST IN int& height,
	CONST IN int& lenght,
	CONST IN float* costSegments,
	CONST IN float* pixelCost,
	CONST IN float* turboPixelImage,
	OUT float* depth)
{
	GetDepthMap_GPU << <gridSize, blockSize >> >
		(width, height, lenght,
			costSegments,
			pixelCost,
			turboPixelImage,
			depth);
}