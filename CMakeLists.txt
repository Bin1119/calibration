cmake_minimum_required(VERSION 3.10)
project(Calibration)

# 找opencv库
find_package(OpenCV REQUIRED)

# 链接opnecv
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE release)

add_executable(Calibration calibrate.cpp)

# 将库文件链接到可执行程序上
target_link_libraries( Calibration  ${OpenCV_LIBS})
