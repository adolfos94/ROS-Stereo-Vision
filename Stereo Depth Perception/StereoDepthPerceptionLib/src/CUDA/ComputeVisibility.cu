#include "Wrapper.h"

__global__ void ComputeVisibility_GPU(
	CONST IN int width,
	CONST IN int height,
	CONST IN int lenght,
	CONST IN float* costCoordinatesLeft,
	CONST IN float* costCoordinatesRight,
	CONST IN float* turboPixelImageLeft,
	CONST IN float* turboPixelImageRight,
	OUT float* costSegmentsLeft,
	OUT float* costSegmentsRight,
	OUT float* disparityLeft,
	OUT float* disparityRight)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idxSegment;
	int idxCoords;

	if (idx >= 0 && idx < lenght)
	{
		int minDisparityLeft = 0;
		int minDisparityRight = 0;

		float minSegmentCostLeft = FLT_MAX;
		float minSegmentCostRight = FLT_MAX;

		for (int d = 0; d < MaxDisparity; ++d)
		{
			idxSegment = (idx * MaxDisparity) + d;

			if (costSegmentsLeft[idxSegment] < minSegmentCostLeft)
			{
				minSegmentCostLeft = costSegmentsLeft[idxSegment];
				minDisparityLeft = d;
			}
			if (costSegmentsRight[idxSegment] < minSegmentCostRight)
			{
				minSegmentCostRight = costSegmentsRight[idxSegment];
				minDisparityRight = d;
			}
		}

		disparityLeft[idx] = minDisparityLeft;
		disparityRight[idx] = minDisparityRight;

		int idxX; int idxY;
		int idxLeft; int idxRight;

		idxY = costCoordinatesLeft[(idx * CostParams) + 0] + 0.5 - minDisparityLeft;
		idxX = costCoordinatesLeft[(idx * CostParams) + 1] + 0.5;

		if (idxY >= 0 && idxY < width)
		{
			idxLeft = turboPixelImageLeft[idxX * width + idxY];
			if (abs(minDisparityLeft - disparityRight[idxLeft]) > 1)
			{
				for (int d = 0; d < MaxDisparity; ++d)
				{
					idxSegment = (idx * MaxDisparity) + d;
					costSegmentsLeft[idxSegment] = 0;
				}
			}
		}

		idxY = costCoordinatesRight[(idx * CostParams) + 0] + 0.5 + minDisparityRight;
		idxX = costCoordinatesRight[(idx * CostParams) + 1] + 0.5;

		if (idxY >= 0 && idxY < width)
		{
			idxRight = turboPixelImageRight[idxX * width + idxY];
			if (abs(minDisparityRight - disparityLeft[idxRight]) > 1)
			{
				for (int d = 0; d < MaxDisparity; ++d)
				{
					idxSegment = (idx * MaxDisparity) + d;
					costSegmentsRight[idxSegment] = 0;
				}
			}
		}
	}
}

VOID CUDA::ComputeVisibility(
	CONST IN dim3& gridSize,
	CONST IN dim3& blockSize,
	CONST IN int& width,
	CONST IN int& height,
	CONST IN int& lenght,
	CONST IN float* costCoordinatesLeft,
	CONST IN float* costCoordinatesRight,
	CONST IN float* turboPixelImageLeft,
	CONST IN float* turboPixelImageRight,
	OUT float* costSegmentsLeft,
	OUT float* costSegmentsRight,
	OUT float* disparityLeft,
	OUT float* disparityRight)
{
	ComputeVisibility_GPU << <gridSize, blockSize >> >
		(width, height, lenght,
			costCoordinatesLeft, costCoordinatesRight,
			turboPixelImageLeft, turboPixelImageRight,
			costSegmentsLeft, costSegmentsRight,
			disparityLeft, disparityRight);
}