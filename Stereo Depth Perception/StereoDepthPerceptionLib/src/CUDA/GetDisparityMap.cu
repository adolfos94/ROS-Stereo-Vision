#include "Wrapper.h"

__global__ void GetDisparityMap_GPU(
	CONST IN int lenght,
	CONST IN float* costSegments,
	OUT float* disparity)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idxSegment;

	if (idx >= 0 && idx < lenght)
	{
		float dp = 0;
		float minCost = FLT_MAX;

		for (int d = 0; d < MaxDisparity; ++d)
		{
			idxSegment = (idx * MaxDisparity) + d;

			if (costSegments[idxSegment] < minCost)
			{
				minCost = costSegments[idxSegment];
				dp = d;
			}
		}

		disparity[idx] = dp;
	}
}

VOID CUDA::GetDisparityMap(
	CONST IN dim3& gridSize,
	CONST IN dim3& blockSize,
	CONST IN int& lenght,
	CONST IN float* costSegments,
	OUT float* disparity)
{
	GetDisparityMap_GPU << <gridSize, blockSize >> >
		(lenght, costSegments, disparity);
}