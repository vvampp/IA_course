cmake_minimum_required(VERSION 3.18)
project(MinimumDistanceClassifier LANGUAGES CXX CUDA)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# change build CUDA support
option(USE_CUDA "Build with CUDA support" ON)

# include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# compilation flags C++
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")

# compilation flags CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Ada (maybe upgrade CUDA tookit ...)
set(CMAKE_CUDA_ARCHITECTURES "89")

# sources
set(CPP_SOURCES
  src/classifier.cpp
)

set(CUDA_SOURCES
  src/cuda_kernels.cu
)

# Static lib 
if(USE_CUDA)
  add_definitions(-DUSE_CUDA)
    
  add_library(mdc_lib STATIC ${CPP_SOURCES} ${CUDA_SOURCES})
    
  set_target_properties(mdc_lib PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      POSITION_INDEPENDENT_CODE ON
  )
    
  target_link_libraries(mdc_lib PUBLIC CUDA::cudart)
else()
  add_library(mdc_lib STATIC ${CPP_SOURCES})
endif()


# build info
message(STATUS "=== Build Configuration ===")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "USE_CUDA: ${USE_CUDA}")
message(STATUS "===========================")
