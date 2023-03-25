cmake_minimum_required(VERSION 3.14)
project(SS_Assignement)

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")


add_subdirectory(src)
add_subdirectory(src/OMP)
add_subdirectory(src/CUDA)