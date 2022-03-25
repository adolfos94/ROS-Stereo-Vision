#include "Stereo Depth Perception.h"
#include "../src/CUDA/Wrapper.h"

float* StereoDepthPerception::PenaltyMatrix_GPU = nullptr;

float* StereoDepthPerception::LeftImage_GPU = nullptr;
float* StereoDepthPerception::RightImage_GPU = nullptr;

float* StereoDepthPerception::GrayLeftImage_GPU = nullptr;
float* StereoDepthPerception::GrayRightImage_GPU = nullptr;

float* StereoDepthPerception::GxLeft_GPU = nullptr;
float* StereoDepthPerception::GyLeft_GPU = nullptr;
float* StereoDepthPerception::GxRight_GPU = nullptr;
float* StereoDepthPerception::GyRight_GPU = nullptr;

float* StereoDepthPerception::PixelCostLeft_GPU = nullptr;
float* StereoDepthPerception::PixelCostRight_GPU = nullptr;

float* StereoDepthPerception::TurboPixelLeft_GPU = nullptr;
float* StereoDepthPerception::TurboPixelRight_GPU = nullptr;

float* StereoDepthPerception::CostCoordinatesLeft_GPU = nullptr;
float* StereoDepthPerception::CostCoordinatesRight_GPU = nullptr;

float* StereoDepthPerception::CostSegmentsLeft_GPU = nullptr;
float* StereoDepthPerception::CostSegmentsRight_GPU = nullptr;

float* StereoDepthPerception::GraphLeft_GPU = nullptr;
float* StereoDepthPerception::GraphRight_GPU = nullptr;
float* StereoDepthPerception::IntensitiesLeft_GPU = nullptr;
float* StereoDepthPerception::IntensitiesRight_GPU = nullptr;
float* StereoDepthPerception::NeighborhoodsLeft_GPU = nullptr;
float* StereoDepthPerception::NeighborhoodsRight_GPU = nullptr;

float* StereoDepthPerception::IterativeCostLeft_GPU = nullptr;
float* StereoDepthPerception::IterativeCostRight_GPU = nullptr;
float* StereoDepthPerception::InitialCostLeft_GPU = nullptr;
float* StereoDepthPerception::InitialCostRight_GPU = nullptr;

float* StereoDepthPerception::DisparityLeft_GPU = nullptr;
float* StereoDepthPerception::DisparityRight_GPU = nullptr;
float* StereoDepthPerception::Disparity_GPU = nullptr;
float* StereoDepthPerception::Depth_GPU = nullptr;

int StereoDepthPerception::Width = 0;
int StereoDepthPerception::Height = 0;
int StereoDepthPerception::Lenght = 0;
int StereoDepthPerception::NumberGridSegments = 0;
size_t StereoDepthPerception::DataLenght = 0;

int StereoDepthPerception::MaxThreadsPerBlock = 0;
dim3 StereoDepthPerception::BlockSize, StereoDepthPerception::blockSize;
dim3 StereoDepthPerception::GridSize, StereoDepthPerception::gridSize;

VOID SaveMAT2TXT(
	CONST IN cv::Mat& image,
	CONST IN std::string& str)
{
	std::ofstream file;
	std::stringstream name;
	name << str << ".txt";
	file.open(name.str(), std::ofstream::out);

	int i, j;
	for (i = 0; i < image.rows; ++i)
	{
		for (j = 0; j < image.cols; ++j)
		{
			file << std::setprecision(5) << (float)image.at<float>(i, j) << TAB;
		}

		file << ENDL;
	}

	file.close();
}

VOID StereoDepthPerception::CUDA2MAT(
	CONST IN float* image_GPU,
	CONST IN int& width,
	CONST IN int& height,
	OUT cv::Mat& image,
	bool disp)
{
	size_t dataLength = SIZE_PTR(float, width, height);

	float* image_CPU = new float[dataLength];
	cudaMemcpy(image_CPU, image_GPU, dataLength, cudaMemcpyDeviceToHost);
	image = cv::Mat(height, width, CV_32F);
	std::memcpy(image.data, image_CPU, dataLength);

	if (disp)image.convertTo(image, CV_8UC1);

	delete[]image_CPU;
}

VOID StereoDepthPerception::MAT2CUDA(
	CONST IN cv::Mat& image,
	OUT float** image_GPU)
{
	size_t matDataLength = SIZE_PTR(float, image.rows, image.cols) * image.channels();

	cudaMalloc(image_GPU, matDataLength);
	cudaMemcpy(*image_GPU, image.data, matDataLength, cudaMemcpyHostToDevice);
}

VOID StereoDepthPerception::CropImages(
	CONST IN cv::Mat& leftImage,
	CONST IN cv::Mat& rightImage)
{
	cv::Mat leftCroppedImage;
	leftImage(cv::Rect(0, 0, Width, Height)).copyTo(leftCroppedImage);
	cv::Mat rightCroppedImage;
	rightImage(cv::Rect(0, 0, Width, Height)).copyTo(rightCroppedImage);

	MAT2CUDA(leftCroppedImage, &LeftImage_GPU);
	MAT2CUDA(rightCroppedImage, &RightImage_GPU);
}

VOID StereoDepthPerception::ConvertRGB2GrayScale(CONST IN int channels)
{
	CUDA::ConvertRGB2GrayScale(
		GridSize, BlockSize, Width, Height, Lenght,
		channels, LeftImage_GPU, GrayLeftImage_GPU);
	CUDA::ConvertRGB2GrayScale(
		GridSize, BlockSize, Width, Height, Lenght,
		channels, RightImage_GPU, GrayRightImage_GPU);

	cudaDeviceSynchronize();
}

VOID StereoDepthPerception::GradientOperations()
{
	CUDA::GradientConvolution(
		GridSize, BlockSize, Width, Height, Lenght,
		GrayLeftImage_GPU, GxLeft_GPU, GyLeft_GPU);

	CUDA::GradientConvolution(
		GridSize, BlockSize, Width, Height, Lenght,
		GrayRightImage_GPU, GxRight_GPU, GyRight_GPU);

	cudaDeviceSynchronize();
}

VOID StereoDepthPerception::GradientMatching()
{
	CUDA::GradientMatching(
		GridSize, BlockSize, Width, Height, Lenght,
		GxLeft_GPU, GyLeft_GPU, GxRight_GPU, GyRight_GPU, PixelCostLeft_GPU, PixelCostRight_GPU);

	cudaDeviceSynchronize();
}

VOID StereoDepthPerception::TurboPixelGeneration()
{
	CUDA::TurboPixelGeneration(
		GridSize, BlockSize, Width, Height, Lenght,
		GrayLeftImage_GPU, TurboPixelLeft_GPU);

	CUDA::TurboPixelGeneration(
		GridSize, BlockSize, Width, Height, Lenght,
		GrayRightImage_GPU, TurboPixelRight_GPU);

	cudaDeviceSynchronize();
}

VOID StereoDepthPerception::TurboPixelClustering()
{
	CUDA::TurboPixelClustering(
		GridSize, BlockSize, Width, Height, Lenght,
		GrayLeftImage_GPU, TurboPixelLeft_GPU);

	CUDA::TurboPixelClustering(
		GridSize, BlockSize, Width, Height, Lenght,
		GrayRightImage_GPU, TurboPixelRight_GPU);

	cudaDeviceSynchronize();
}

VOID StereoDepthPerception::TurboPixelExpansion(CONST IN int iterations)
{
	for (int iter = 0; iter < iterations; ++iter)
	{
		CUDA::TurboPixelExpansion(
			GridSize, BlockSize, Width, Height, Lenght,
			GrayLeftImage_GPU, TurboPixelLeft_GPU);

		CUDA::TurboPixelExpansion(
			GridSize, BlockSize, Width, Height, Lenght,
			GrayRightImage_GPU, TurboPixelRight_GPU);

		cudaDeviceSynchronize();
	}
}

VOID StereoDepthPerception::ComputeCostCoordinates()
{
	CUDA::ComputeCostCoordinates(
		GridSize, BlockSize, Width, Height, Lenght,
		GrayLeftImage_GPU, TurboPixelLeft_GPU, CostCoordinatesLeft_GPU);

	CUDA::ComputeCostCoordinates(
		GridSize, BlockSize, Width, Height, Lenght,
		GrayRightImage_GPU, TurboPixelRight_GPU, CostCoordinatesRight_GPU);

	cudaDeviceSynchronize();
}

VOID StereoDepthPerception::ComputeCostSegments()
{
	CUDA::ComputeCostSegments(
		GridSize, BlockSize, Width, Height, Lenght,
		PixelCostLeft_GPU, TurboPixelLeft_GPU, CostSegmentsLeft_GPU);

	CUDA::ComputeCostSegments(
		GridSize, BlockSize, Width, Height, Lenght,
		PixelCostRight_GPU, TurboPixelRight_GPU, CostSegmentsRight_GPU);

	cudaDeviceSynchronize();
}

VOID StereoDepthPerception::ComputeCostNormalization()
{
	CUDA::ComputeCostNormalization(
		gridSize, blockSize,
		NumberGridSegments, CostCoordinatesLeft_GPU, CostSegmentsLeft_GPU);

	CUDA::ComputeCostNormalization(
		gridSize, blockSize,
		NumberGridSegments, CostCoordinatesRight_GPU, CostSegmentsRight_GPU);

	cudaDeviceSynchronize();
}

VOID StereoDepthPerception::GraphConstruction()
{
	CUDA::GraphConstruction(
		GridSize, BlockSize, Width, Height, NumberGridSegments,
		GrayLeftImage_GPU, TurboPixelLeft_GPU, GraphLeft_GPU, IntensitiesLeft_GPU);

	CUDA::GraphConstruction(
		GridSize, BlockSize, Width, Height, NumberGridSegments,
		GrayRightImage_GPU, TurboPixelRight_GPU, GraphRight_GPU, IntensitiesRight_GPU);

	cudaDeviceSynchronize();
}

VOID StereoDepthPerception::NeighborhoodsConstruction()
{
	CUDA::NeighborhoodsConstruction(
		gridSize, blockSize, NumberGridSegments,
		GraphLeft_GPU, IntensitiesLeft_GPU, NeighborhoodsLeft_GPU);

	CUDA::NeighborhoodsConstruction(
		gridSize, blockSize, NumberGridSegments,
		GraphRight_GPU, IntensitiesRight_GPU, NeighborhoodsRight_GPU);

	cudaDeviceSynchronize();
}

VOID StereoDepthPerception::RandomWalkWithRestart(CONST IN int iterations)
{
	size_t segmentDataLenght = SIZE_PTR(float, MaxDisparity, NumberGridSegments);

	cudaMemcpy(IterativeCostLeft_GPU, CostSegmentsLeft_GPU, segmentDataLenght, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
	cudaMemcpy(IterativeCostRight_GPU, CostSegmentsRight_GPU, segmentDataLenght, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
	cudaMemcpy(InitialCostLeft_GPU, CostSegmentsLeft_GPU, segmentDataLenght, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
	cudaMemcpy(InitialCostRight_GPU, CostSegmentsRight_GPU, segmentDataLenght, cudaMemcpyKind::cudaMemcpyDeviceToDevice);

	// InitialCost = RestartEpsilon * InitialCost
	CUBLAS::ScalarMatrixMultiplication(RestartEpsilon, NumberGridSegments, MaxDisparity, InitialCostLeft_GPU);
	CUBLAS::ScalarMatrixMultiplication(RestartEpsilon, NumberGridSegments, MaxDisparity, InitialCostRight_GPU);

	for (int i = 0; i < iterations; ++i)
	{
		RestartProbabilities();
		ComputeVisibility();
		ComputeFidelity();
	}
}

/// <summary>
/// CostSegments = ((1 - RestartEpsilon) * Graph * IterativeCost) + InitialCost
/// </summary>
VOID StereoDepthPerception::RestartProbabilities()
{
	// CostSegments = Graph * IterativeCost
	CUBLAS::MatrixMultiplication(GraphLeft_GPU, IterativeCostLeft_GPU, NumberGridSegments, NumberGridSegments, MaxDisparity, CostSegmentsLeft_GPU);
	CUBLAS::MatrixMultiplication(GraphRight_GPU, IterativeCostRight_GPU, NumberGridSegments, NumberGridSegments, MaxDisparity, CostSegmentsRight_GPU);

	// CostSegments = (1 - RestartEpsilon) * CostSegments
	CUBLAS::ScalarMatrixMultiplication(1 - RestartEpsilon, NumberGridSegments, MaxDisparity, CostSegmentsLeft_GPU);
	CUBLAS::ScalarMatrixMultiplication(1 - RestartEpsilon, NumberGridSegments, MaxDisparity, CostSegmentsRight_GPU);

	// CostSegments = InitialCost + CostSegments
	CUBLAS::MatrixAddition(InitialCostLeft_GPU, NumberGridSegments, MaxDisparity, CostSegmentsLeft_GPU);
	CUBLAS::MatrixAddition(InitialCostRight_GPU, NumberGridSegments, MaxDisparity, CostSegmentsRight_GPU);

	cudaDeviceSynchronize();
}

VOID StereoDepthPerception::ComputeVisibility()
{
	CUDA::ComputeVisibility(
		gridSize, blockSize, Width, Height, NumberGridSegments,
		CostCoordinatesLeft_GPU, CostCoordinatesRight_GPU,
		TurboPixelLeft_GPU, TurboPixelRight_GPU,
		CostSegmentsLeft_GPU, CostSegmentsRight_GPU,
		DisparityLeft_GPU, DisparityRight_GPU);

	cudaDeviceSynchronize();
}

VOID StereoDepthPerception::ComputeFidelity()
{
	CUDA::ComputeFidelity(
		gridSize, blockSize, NumberGridSegments,
		PenaltyMatrix_GPU,
		NeighborhoodsLeft_GPU, NeighborhoodsRight_GPU,
		CostSegmentsLeft_GPU, CostSegmentsRight_GPU,
		DisparityLeft_GPU, DisparityRight_GPU);

	cudaDeviceSynchronize();
}

VOID StereoDepthPerception::GetDisparityMap()
{
	CUDA::GetDisparityMap(
		gridSize, blockSize, NumberGridSegments,
		CostSegmentsLeft_GPU,
		Disparity_GPU);

	cudaDeviceSynchronize();
}

VOID StereoDepthPerception::GetDepthMap()
{
	CUDA::GetDepthMap(
		GridSize, BlockSize, Width, Height, NumberGridSegments,
		CostSegmentsLeft_GPU,
		PixelCostLeft_GPU,
		TurboPixelLeft_GPU,
		Depth_GPU);

	cudaDeviceSynchronize();
}

VOID StereoDepthPerception::SetPenaltyMatrix()
{
	cv::Mat penalty = cv::Mat::zeros(MaxDisparity, MaxDisparity, CV_32F);

	int i, j;
	float diff;

	for (i = 0; i < MaxDisparity; ++i)
	{
		for (j = 0; j < MaxDisparity; ++j)
		{
			diff = abs(i - j);
			penalty.at<float>(i, j) = diff < TAU ?
				(diff * diff / PSI / PSI) :
				(TAU * TAU / PSI / PSI);
		}
	}

	MAT2CUDA(penalty, &PenaltyMatrix_GPU);
}

// Allocate [OUT] Memory in GPU
VOID StereoDepthPerception::SetMemoryAllocations()
{
	// GrayScale Images
	cudaMalloc(&GrayLeftImage_GPU, DataLenght);
	cudaMalloc(&GrayRightImage_GPU, DataLenght);

	// Gradient Images
	cudaMalloc(&GxLeft_GPU, DataLenght);
	cudaMalloc(&GyLeft_GPU, DataLenght);
	cudaMalloc(&GxRight_GPU, DataLenght);
	cudaMalloc(&GyRight_GPU, DataLenght);

	// Pixel Cost Matrices
	size_t pixelCostDataLenght = DataLenght * MaxDisparity;

	cudaMalloc(&PixelCostLeft_GPU, pixelCostDataLenght);
	cudaMalloc(&PixelCostRight_GPU, pixelCostDataLenght);

	// Turbo Pixel Images
	cudaMalloc(&TurboPixelLeft_GPU, DataLenght);
	cudaMalloc(&TurboPixelRight_GPU, DataLenght);

	// Cost Params Coordinates Matrices
	size_t costParamsDataLenght = sizeof(float) * CostParams * NumberGridSegments;

	cudaMalloc(&CostCoordinatesLeft_GPU, costParamsDataLenght);
	cudaMalloc(&CostCoordinatesRight_GPU, costParamsDataLenght);

	// Cost Segments Matrices
	size_t segmentDataLenght = sizeof(float) * MaxDisparity * NumberGridSegments;

	cudaMalloc(&CostSegmentsLeft_GPU, segmentDataLenght);
	cudaMalloc(&CostSegmentsRight_GPU, segmentDataLenght);

	cudaMalloc(&IterativeCostLeft_GPU, segmentDataLenght);
	cudaMalloc(&IterativeCostRight_GPU, segmentDataLenght);
	cudaMalloc(&InitialCostLeft_GPU, segmentDataLenght);
	cudaMalloc(&InitialCostRight_GPU, segmentDataLenght);

	// Graph and Intensities Matrices
	size_t graphDataLenght = sizeof(float) * NumberGridSegments * NumberGridSegments;
	size_t intensitiesDataLenght = sizeof(float) * 3 * NumberGridSegments;

	cudaMalloc(&GraphLeft_GPU, graphDataLenght);
	cudaMalloc(&GraphRight_GPU, graphDataLenght);
	cudaMalloc(&IntensitiesLeft_GPU, intensitiesDataLenght);
	cudaMalloc(&IntensitiesRight_GPU, intensitiesDataLenght);

	// Neighborhoods Matrices
	size_t neighborhoodsDataLenght = sizeof(float) * Neighborhoods * NumberGridSegments;

	cudaMalloc(&NeighborhoodsLeft_GPU, neighborhoodsDataLenght);
	cudaMalloc(&NeighborhoodsRight_GPU, neighborhoodsDataLenght);

	// Disparity Matrices
	size_t disparityDataLength = SIZE_PTR(float, NumberGridSegments, 1);

	cudaMalloc(&DisparityLeft_GPU, disparityDataLength);
	cudaMalloc(&DisparityRight_GPU, disparityDataLength);
	cudaMalloc(&Disparity_GPU, disparityDataLength);

	// Depth Image
	cudaMalloc(&Depth_GPU, DataLenght);
}

VOID StereoDepthPerception::GetDepthImage(OUT cv::Mat& depthMap)
{
	CUDA2MAT(Depth_GPU, Width, Height, depthMap, true);

	double minVal;
	double maxVal;
	cv::minMaxLoc(depthMap, &minVal, &maxVal);

	std::cout << "**************************" << ENDL
		<< "Min Val Depth : " << minVal << ENDL
		<< "Max Val Depth : " << maxVal << ENDL;

	depthMap = (depthMap / maxVal) * 255;
}

VOID StereoDepthPerception::Compute(
	CONST IN cv::Mat& leftImage,
	CONST IN cv::Mat& rightImage)
{
	CropImages(leftImage, rightImage);
	ConvertRGB2GrayScale(leftImage.channels());
	GradientOperations();
	GradientMatching();
	TurboPixelGeneration();
	TurboPixelClustering();
	TurboPixelExpansion();
	ComputeCostCoordinates();
	ComputeCostSegments();
	ComputeCostNormalization();
	GraphConstruction();
	NeighborhoodsConstruction();
	RandomWalkWithRestart();
	GetDisparityMap();
	GetDepthMap();
}

VOID StereoDepthPerception::Setup(
	CONST IN cv::Size size)
{
	int threads;
	CUDA::GetDeviceInfo(MaxThreadsPerBlock);

	threads = floor(sqrt(MaxThreadsPerBlock));

	Width = size.width - (size.width % TurboPixelSize);
	Height = size.height - (size.height % TurboPixelSize);
	Lenght = Width * Height;
	NumberGridSegments = Lenght / (TurboPixelSize * TurboPixelSize);

	DataLenght = sizeof(float) * Lenght;

	BlockSize = dim3(threads, threads, 1);
	GridSize = dim3(ceil((float)Height / threads), ceil((float)Width / threads), 1);

	blockSize = dim3(MaxThreadsPerBlock, 1, 1);
	gridSize = dim3(ceil((float)NumberGridSegments / MaxThreadsPerBlock), 1, 1);

	SetMemoryAllocations();
	SetPenaltyMatrix();
}