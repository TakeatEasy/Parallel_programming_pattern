cmake_minimum_required(VERSION 3.10)

project(PARALLEL_PROGRAMMING_PATTERN)

enable_language(CUDA)


include_directories(${CMAKE_SOURCE_DIR}/utils)

add_executable(my_test ${CMAKE_SOURCE_DIR}/test/test_generator.cpp)
add_executable(cuda_test ${CMAKE_SOURCE_DIR}/reduce/reduce_basic.cu)
add_executable(cuda_reduce_baseline ${CMAKE_SOURCE_DIR}/reduce/reduce_baseline.cu)