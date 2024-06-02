cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME "jax_plate_lib")
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -g -mtune=native -march=native")

find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

message("WARNING: If something goes wrong with suitesparse change the dirs in plate_inverse_problem/source/jax_plate_lib/CMakeLists.txt:12-13")
set(SUITESPARSE_INCLUDE "/usr/include/suitesparse")
set(SUITESPARSE_LIB "/usr/lib64")

find_path(UMFPACK_INCLUDE NAMES umfpack.h PATHS ${SUITESPARSE_INCLUDE} REQUIRED)
find_library(UMFPACK NAMES umfpack PATHS ${SUITESPARSE_LIB} REQUIRED)

set(SOURCES "src")

pybind11_add_module(${PROJECT_NAME} ${SOURCES}/main.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${UMFPACK_INCLUDE})
target_include_directories(${PROJECT_NAME} PRIVATE include)


target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(${PROJECT_NAME} PUBLIC ${UMFPACK})

message(STATUS "Found UMFPACK library here: ${UMFPACK} and headers here: ${UMFPACK_INCLUDE}")
