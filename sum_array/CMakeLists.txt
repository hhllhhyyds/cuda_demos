add_executable(sum_array
    sum_array.cu
)
target_link_libraries(sum_array
    timer
)
target_include_directories(sum_array PRIVATE
    ${CMAKE_SOURCE_DIR}
)
add_test(
    NAME cuda_sum_array
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/sum_array${CMAKE_EXECUTABLE_SUFFIX}
)

add_executable(cublas_sum
    cublas_sum.cu
)
target_link_libraries(cublas_sum
    cublas
)
target_include_directories(cublas_sum PRIVATE
    ${CMAKE_SOURCE_DIR}
)
add_test(
    NAME cublas_sum
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/cublas_sum${CMAKE_EXECUTABLE_SUFFIX}
)