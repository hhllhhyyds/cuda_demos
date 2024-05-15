cmake_minimum_required(VERSION 3.25)

project(CUDA_DEMOS LANGUAGES C CXX CUDA)

include(CTest)
enable_testing()

set(CUDA_INCLUDE_DIR ${CUDAToolkit_ROOT}/include)

add_subdirectory(sum_array)
add_subdirectory(timer)
add_subdirectory(gpu_device_info)
add_subdirectory(sum_matrix)
