
ADD_LIBRARY(
	StereoDepthPerceptionLib
	"plugin.cpp"
	"src/Stereo Depth Perception.cpp"
	"src/CUDA/DeviceInfo.cu"
	"src/CUDA/CublasMatrixOperations.cu"
	"src/CUDA/GrayScale.cu"
	"src/CUDA/GradientConv.cu"
	"src/CUDA/GradientMatching.cu"
	"src/CUDA/TurboPixelGeneration.cu"
	"src/CUDA/TurboPixelClustering.cu"
	"src/CUDA/TurboPixelExpansion.cu"
	"src/CUDA/ComputeCostCoordinates.cu"
	"src/CUDA/ComputeCostSegments.cu"
	"src/CUDA/ComputeCostNormalization.cu"
	"src/CUDA/GraphConstruction.cu"
	"src/CUDA/NeighborhoodsConstruction.cu"
	"src/CUDA/ComputeVisibility.cu"
	"src/CUDA/ComputeFidelity.cu"
	"src/CUDA/GetDisparityMap.cu"
	"src/CUDA/GetDepthMap.cu"
)