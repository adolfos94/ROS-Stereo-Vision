cmake_minimum_required(VERSION 3.10)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)

PROJECT(cameracalib)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS})

MACRO(add_example name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS})
ENDMACRO()

add_example(GetCalibrationImages)
add_example(GetCalibrationMatrices)
add_example(GetRectifiedImages)

