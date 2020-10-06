cmake_minimum_required(VERSION 3.16)
project(cpp_benchmarks)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

set(BENCHMARK_ENABLE_TESTING off)
add_subdirectory(benchmark)

add_executable(cpp_benchmarks main.cpp)
target_link_libraries(cpp_benchmarks benchmark::benchmark)