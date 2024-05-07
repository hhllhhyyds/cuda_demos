cmake_minimum_required(VERSION 3.25)

project(CUDA_DEMOS LANGUAGES CXX CUDA)

include(CTest)

set(CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(sum_array)
add_subdirectory(timer)
