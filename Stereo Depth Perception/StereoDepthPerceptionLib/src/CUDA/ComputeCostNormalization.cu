#include "Wrapper.h"

__global__ void ComputeCostNormalization_GPU(
	CONST IN int lenght,
	OUT float* costCoordinates,
	OUT float* costSegments)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx >= 0 && idx < lenght)
	{
		int total = costCoordinates[(idx * CostParams) + 3];

		if (total)
		{
			costCoordinates[(idx * CostParams) + 0] /= total;
			costCoordinates[(idx * CostParams) + 1] /= total;
			costCoordinates[(idx * CostParams) + 2] /= total;
		}

		for (int d = 0; d < MaxDisparity; ++d)
		{
			if (total)
			{
				costSegments[(idx * MaxDisparity) + d] /= total;
			}
		}
	}
}

VOID CUDA::ComputeCostNormalization(
	CONST IN dim3& gridSize,
	CONST IN dim3& blockSize,
	CONST IN int& lenght,
	OUT float* costCoordinates,
	OUT float* costSegments)
{
	ComputeCostNormalization_GPU << <gridSize, blockSize >> > (lenght, costCoordinates, costSegments);
}