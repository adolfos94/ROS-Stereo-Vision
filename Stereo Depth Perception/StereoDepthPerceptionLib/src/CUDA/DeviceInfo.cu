#include "Wrapper.h"

VOID GetDeviceInfo_GPU(OUT int& maxThreadsPerBlock)
{
	int driverVersion;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	cudaDriverGetVersion(&driverVersion);
	std::cout << "**************************" << ENDL;
	printf("\nDevice name: %s\n", prop.name);
	printf("  Memory Clock Rate (KHz): %d\n",
		prop.memoryClockRate);
	printf("  Memory Bus Width (bits): %d\n",
		prop.memoryBusWidth);
	printf("  Peak Memory Bandwidth (GB/s): %f\n",
		2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
	printf("  Max Threads per Block: %d\n",
		prop.maxThreadsPerBlock);

	maxThreadsPerBlock = prop.maxThreadsPerBlock;
}

VOID CUDA::GetDeviceInfo(OUT int& maxThreadsPerBlock)
{
	GetDeviceInfo_GPU(maxThreadsPerBlock);
}