cmake_minimum_required(VERSION 3.0)
project(opencv1)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV)

#include_directories(${opencv_INCLUDE_DIRS})

add_executable(opencv1 main.cpp)
target_link_libraries(opencv1 ${OpenCV_LIBRARIES})

