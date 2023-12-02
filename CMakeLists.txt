cmake_minimum_required(VERSION 3.10)

project(PARALLEL_PROGRAMMING_PATTERN)

enable_language(CUDA)


include_directories(${CMAKE_SOURCE_DIR}/utils)

# number 86 is for 3060
# set(CUDA_NVCC_FLAGS -arch=compute_86;-code=compute_86)

add_executable(my_test ${CMAKE_SOURCE_DIR}/test/test_generator.cpp)
add_executable(cuda_test ${CMAKE_SOURCE_DIR}/reduce/reduce_basic.cu)
add_executable(cuda_reduce_baseline ${CMAKE_SOURCE_DIR}/reduce/reduce_baseline.cu)
add_executable(reduce_warp_divergence ${CMAKE_SOURCE_DIR}/reduce/reduce_warp_divergence.cu)


# for sgemm
add_executable(sgemm_baseline ${CMAKE_SOURCE_DIR}/sgemm/sgemm_baseline.cu)