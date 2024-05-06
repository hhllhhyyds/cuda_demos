add_executable(sum_array
    sum_array.cu
)
add_test(
    NAME cuda_sum_array
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/sum_array${CMAKE_EXECUTABLE_SUFFIX}
)