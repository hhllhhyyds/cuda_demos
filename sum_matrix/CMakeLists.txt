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