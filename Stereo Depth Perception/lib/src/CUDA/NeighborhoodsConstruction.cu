#include "Wrapper.h"

__global__ void NeighborhoodsConstruction_GPU(
	CONST IN int lenght,
	OUT float* graph,
	OUT float* intensities,
	OUT float* neighborhoods)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idxGraph, idxIntensities, idxNeighborhoods;

	int count = 0;

	if (idx >= 0 && idx < lenght)
	{
		idxNeighborhoods = idx * Neighborhoods;
		idxIntensities = (idx * 3) + 2;

		for (int k = 0; k < lenght; ++k)
		{
			idxGraph = idx + (k * lenght);

			if (graph[idxGraph] != 0)
			{
				neighborhoods[idxNeighborhoods] += 1;
				count = neighborhoods[idxNeighborhoods];
				neighborhoods[idxNeighborhoods + (count * 2) - 1] = k;
				neighborhoods[idxNeighborhoods + (count * 2)] = graph[idxGraph];
			}
		}

		for (int k = 0; k < lenght; ++k)
		{
			idxGraph = idx + (k * lenght);

			intensities[idxIntensities] += graph[idxGraph];
		}

		__syncthreads();

		for (int k = 0; k < lenght; ++k)
		{
			idxGraph = idx + (k * lenght);

			graph[idxGraph] = !intensities[idxIntensities]
				? 0
				: graph[idxGraph] / intensities[idxIntensities];
		}
	}
}

VOID CUDA::NeighborhoodsConstruction(
	CONST IN dim3& gridSize,
	CONST IN dim3& blockSize,
	CONST IN int& lenght,
	OUT float* graph,
	OUT float* intensities,
	OUT float* neighborhoods)
{
	NeighborhoodsConstruction_GPU << <gridSize, blockSize >> > (lenght, graph, intensities, neighborhoods);
}