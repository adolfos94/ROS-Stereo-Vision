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
ADD_SUBDIRECTORY("lib")

# Add source to this project's executable.
ADD_EXECUTABLE(StereoDepthPerception 
	"main.cpp"
)

message(STATUS "CUDA LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "CUDA CUBLAS LIBRARIES: ${CUDA_CUBLAS_LIBRARIES}")

TARGET_LINK_LIBRARIES(StereoDepthPerception ${OpenCV_LIBS}) 
TARGET_LINK_LIBRARIES(StereoDepthPerception "StereoDepthPerceptionLib" ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

# Post-build event

if(UNIX AND NOT APPLE)
    # Linux, BSD, Solaris, Minix
    set(LIB_NAME "StereoDepthPerceptionLib.a")
    set(EXTERNAL_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/../src/depth_perception_sub/libraries/${LIB_NAME}")
    add_custom_command(TARGET StereoDepthPerception POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:StereoDepthPerceptionLib> ${EXTERNAL_LIBS}
        COMMENT "Created ${EXTERNAL_LIBS}")
endif()

