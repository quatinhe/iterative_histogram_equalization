cmake_minimum_required(VERSION 3.16)
project(project)

set(CMAKE_CXX_STANDARD 17)

#find OpenMP
find_package(OpenMP REQUIRED)

###############
# libWB
###############

set(BUILD_LIBWB_LIBRARY ON CACHE BOOL "Turn on build of libwb library")
add_subdirectory(libwb)
include_directories(libwb)

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
)

target_link_libraries(project_par_gpu PUBLIC OpenMP::OpenMP_CXX wb ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
set_target_properties(project_par_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(project_par_gpu_metrics
        src/main.cpp
        src/histogram_eq_metrics.cu
        src/histogram_eq_metrics.cu
)

target_compile_options(project_par_gpu_metrics PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v,--verbose>)
target_link_libraries(project_par_gpu_metrics PUBLIC OpenMP::OpenMP_CXX wb ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
set_target_properties(project_par_gpu_metrics PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
