# CMakeList.txt : CMake project for Stereo Depth Perception, include source and define
# project specific logic here.
#
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)
PROJECT ("StereoDepthPerception" LANGUAGES CXX CUDA)

FIND_PACKAGE( OpenCV REQUIRED )                              
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

FIND_PACKAGE(CUDA REQUIRED)
INCLUDE_DIRECTORIES( ${CUDA_INCLUDE_DIRS})

# ADD LIBRARY
ADD_SUBDIRECTORY(StereoDepthPerceptionLib)

# Add source to this project's executable.
ADD_EXECUTABLE(StereoDepthPerception 
	"main.cpp"
    "source/WebCam/WebCam.cpp"
)

message(STATUS "CUDA LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "CUDA CUBLAS LIBRARIES: ${CUDA_CUBLAS_LIBRARIES}")

TARGET_LINK_LIBRARIES(StereoDepthPerception ${OpenCV_LIBS}) 
TARGET_LINK_LIBRARIES(StereoDepthPerception "StereoDepthPerceptionLib" ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
