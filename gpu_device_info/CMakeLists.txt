add_library(gpu_device_info
    gpu_device_info.cc
)
target_include_directories(gpu_device_info PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
)

add_executable(test_gpu_device_info
    test_gpu_device_info.cu
)
target_link_libraries(test_gpu_device_info
    gpu_device_info
)
add_test(
    NAME test_gpu_device_info
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/test_gpu_device_info${CMAKE_EXECUTABLE_SUFFIX}
)

