add_executable(check_thread_index
    check_thread_index.cu
)
target_link_libraries(check_thread_index
    gpu_device_info
)
target_include_directories(check_thread_index PRIVATE
    ${CMAKE_SOURCE_DIR}
)
add_test(
    NAME check_thread_index
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/check_thread_index${CMAKE_EXECUTABLE_SUFFIX}
)

macro(sum_matrix_demo src_name)
    add_executable(${src_name}
        ${src_name}.cu
    )
    target_link_libraries(${src_name}
        gpu_device_info
        timer
    )
    target_include_directories(${src_name} PRIVATE
        ${CMAKE_SOURCE_DIR}
    )
    # add_test(
    #     NAME ${src_name}
    #     COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/${src_name}${CMAKE_EXECUTABLE_SUFFIX}
    # )
endmacro(sum_matrix_demo src_name)

sum_matrix_demo(sum_matrix_2d_grid_2d_block)
sum_matrix_demo(sum_matrix_1d_grid_1d_block)
sum_matrix_demo(sum_matrix_2d_grid_1d_block)