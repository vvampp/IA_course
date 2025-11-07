cmake_minimum_required(VERSION 3.18)
project(MinimumDistanceClassifier LANGUAGES CXX CUDA)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Find pybind11
find_package(pybind11 REQUIRED)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Option
option(USE_CUDA "Build with CUDA support" ON)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Compilation flags C++
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")

# Compilation flags CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CUDA_ARCHITECTURES "89")

# Sources
set(CPP_SOURCES src/classifier.cpp)
set(CUDA_SOURCES src/cuda_kernels.cu)
set(BINDING_SOURCES src/python_bindings.cpp)

if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
    
    # Static library with C++/CUDA code
    add_library(mdc_static STATIC ${CPP_SOURCES} ${CUDA_SOURCES})
    
    set_target_properties(mdc_static PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
    )
    
    target_link_libraries(mdc_static PUBLIC CUDA::cudart)
    
    # pynthon module for linkage
    pybind11_add_module(mdc_core MODULE ${BINDING_SOURCES})
    
    target_link_libraries(mdc_core PRIVATE 
        mdc_static
        CUDA::cudart
        CUDA::cuda_driver
    )
    
    set_target_properties(mdc_core PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
else()
    pybind11_add_module(mdc_core MODULE
        ${BINDING_SOURCES}
        ${CPP_SOURCES}
    )
endif()

# Build info
message(STATUS "=== Build Configuration ===")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "USE_CUDA: ${USE_CUDA}")
message(STATUS "===========================")
