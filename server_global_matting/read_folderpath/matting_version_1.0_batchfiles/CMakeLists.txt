cmake_minimum_required(VERSION 3.0)
project(kde)

set(CMAKE_CXX_STANDARD 11)

############ find OpenCV
find_package(OpenCV)
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(sky_matting main.cpp global-matting/globalmatting.h global-matting/globalmatting.cpp)

target_link_libraries( sky_matting ${OpenCV_LIBS} )
