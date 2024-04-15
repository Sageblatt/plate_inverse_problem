cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME "jax_plate_lib")
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -g")

find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

message("WARNING: If something goes wrong with suitesparse change the dirs in CMakeLists.txt:12-13")
set(SUITESPARSE_INCLUDE "/usr/include/suitesparse")
set(SUITESPARSE_LIB "/usr/lib64")

set(SOURCES src)

pybind11_add_module(${PROJECT_NAME} ${SOURCES}/main.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${SUITESPARSE_INCLUDE})
target_include_directories(${PROJECT_NAME} PRIVATE include)

find_library(UMFPACK NAMES umfpack PATHS SUITESPARSE_LIB)

target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(${PROJECT_NAME} PUBLIC ${UMFPACK})

message(STATUS "Found UMFPACK library here: ${UMFPACK}")

#target_include_directories(pyetfe PUBLIC "include" "kissfft")
