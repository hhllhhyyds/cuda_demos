add_library(timer
    timer.cc
)
target_include_directories(timer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(test_timer
    test_timer.cc
)
target_include_directories(test_timer PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(test_timer
    timer
)
add_test(
    NAME test_timer
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/test_timer${CMAKE_EXECUTABLE_SUFFIX}
)

