cmake_minimum_required(VERSION 3.12)

project(orc C CXX)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenMP REQUIRED)

add_subdirectory(spltrack/calibration/lens_distortion_estimation)