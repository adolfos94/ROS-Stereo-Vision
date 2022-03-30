#include "Wrapper.h"

__global__ void ComputeFidelity_GPU(
	CONST IN int lenght,
	CONST IN float* penaltyMatrix,
	CONST IN float* neighborhoodsLeft,
	CONST IN float* neighborhoodsRight,
	OUT float* costSegmentsLeft,
	OUT float* costSegmentsRight,
	OUT float* disparityLeft,
	OUT float* disparityRight)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idxSegment;
	int idxNeighborhoods;

	int neighborIDXLeft; int neighborIDXRight;
	int neighborDisparityLeft; int neighborDisparityRight;
	int averageDisparityLeft;  int averageDisparityRight;
	float disparityNeighborsLeft; float disparityNeighborsRight;
	float normalizationNeighborsLeft; float normalizationNeighborsRight;
	float diffLeft;  float diffRight;

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

		idxNeighborhoods = idx * Neighborhoods;

		normalizationNeighborsLeft = 1;
		normalizationNeighborsRight = 1;
		averageDisparityLeft = 1;
		averageDisparityRight = 1;

		disparityNeighborsLeft = disparityLeft[idx];

		if (disparityNeighborsLeft < 1)
		{
			disparityNeighborsLeft = 0;
			normalizationNeighborsLeft = 0;
		}

		for (int n = 1; n < Neighborhoods / 2; ++n)
		{
			neighborIDXLeft = neighborhoodsLeft[idxNeighborhoods + (n * 2) - 1];
			diffLeft = neighborhoodsLeft[idxNeighborhoods + (n * 2)];
			neighborDisparityLeft = disparityLeft[neighborIDXLeft];

			if (neighborDisparityLeft > 0)
			{
				disparityNeighborsLeft = disparityNeighborsLeft + (neighborDisparityLeft * diffLeft);
				normalizationNeighborsLeft = normalizationNeighborsLeft + diffLeft;
			}
		}

		if (normalizationNeighborsLeft > 0)
		{
			averageDisparityLeft = round(disparityNeighborsLeft / normalizationNeighborsLeft);
		}

		for (int d = 0; d < MaxDisparity; ++d)
		{
			idxSegment = (idx * MaxDisparity) + d;
			costSegmentsLeft[idxSegment] = 
				(costSegmentsLeft[idxSegment] + penaltyMatrix[(averageDisparityLeft * MaxDisparity) + d]) / 2;
		}

		disparityNeighborsRight = disparityRight[idx];

		if (disparityNeighborsRight < 1)
		{
			disparityNeighborsRight = 0;
			normalizationNeighborsRight = 0;
		}

		for (int n = 1; n < Neighborhoods / 2; ++n)
		{
			neighborIDXRight = neighborhoodsRight[idxNeighborhoods + (n * 2) - 1];
			diffRight = neighborhoodsRight[idxNeighborhoods + (n * 2)];
			neighborDisparityRight = disparityRight[neighborIDXRight];

			if (neighborDisparityRight > 0)
			{
				disparityNeighborsRight = disparityNeighborsRight + (neighborDisparityRight * diffRight);
				normalizationNeighborsRight = normalizationNeighborsRight + diffRight;
			}
		}

		if (normalizationNeighborsRight > 0)
		{
			averageDisparityRight = round(disparityNeighborsRight / normalizationNeighborsRight);
		}

		for (int d = 0; d < MaxDisparity; ++d)
		{
			idxSegment = (idx * MaxDisparity) + d;
			costSegmentsRight[idxSegment] = 
				(costSegmentsRight[idxSegment] + penaltyMatrix[(averageDisparityRight * MaxDisparity) + d]) / 2;
		}
	}
}

VOID CUDA::ComputeFidelity(
	CONST IN dim3& gridSize,
	CONST IN dim3& blockSize,
	CONST IN int& lenght,
	CONST IN float* penaltyMatrix,
	CONST IN float* neighborhoodsLeft,
	CONST IN float* neighborhoodsRight,
	OUT float* costSegmentsLeft,
	OUT float* costSegmentsRight,
	OUT float* disparityLeft,
	OUT float* disparityRight)
{
	ComputeFidelity_GPU << <gridSize, blockSize >> >
		(lenght,
			penaltyMatrix,
			neighborhoodsLeft, neighborhoodsRight,
			costSegmentsLeft, costSegmentsRight,
			disparityLeft, disparityRight);
}