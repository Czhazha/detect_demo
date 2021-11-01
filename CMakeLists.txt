cmake_minimum_required(VERSION 2.6)

project(yolox)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(include/jetson-utils include)
link_directories(/home/nano/Development/detect_demo/lib)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# cudnn
include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)
# tensorrt
include_directories(/usr/local/include/jetson-inference)
link_directories(/usr/local/lib/jetson-inference)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolox video.cpp)
target_link_libraries(yolox nvinfer)
target_link_libraries(yolox cudart)
target_link_libraries(yolox ${OpenCV_LIBS})
target_link_libraries(yolox jetson-utils)

add_definitions(-O2 -pthread)

