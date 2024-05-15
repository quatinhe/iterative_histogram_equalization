cmake_minimum_required(VERSION 3.18)
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.4/bin/nvcc")
project(project LANGUAGES CXX CUDA)
message(STATUS "Your message here")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 50)
message(STATUS "CUDA Architecture set to: ${CMAKE_CUDA_ARCHITECTURES}")

find_package(CUDA REQUIRED)


###############
# libWB
###############

set(BUILD_LIBWB_LIBRARY ON CACHE BOOL "Turn on build of libwb library")
add_subdirectory(libwb)
include_directories(libwb)
include_directories(${CUDA_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(include)

###############
# sequencial
###############

include_directories(include)

add_executable(project
        src/main.cpp
        src/histogram_eq.cpp
)
target_link_libraries(project wb)


###############
# paralelo
###############

find_package(OpenMP REQUIRED)

add_executable(project_par
        src/main.cpp
        src/histogram_eq_par.cpp
)

target_link_libraries(project_par PUBLIC OpenMP::OpenMP_CXX wb)

###############
#GPU
###############
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
add_executable(project_par_gpu
        src/main.cpp
        src/histogram_eq.cu
        # Change to .cu if using CUDA kernels
)

target_link_libraries(project_par_gpu PUBLIC OpenMP::OpenMP_CXX wb ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
set_target_properties(project_par_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
