cmake_minimum_required(VERSION 3.5)
project(MinimumDistanceClassifier LANGUAGES CXX)

# Options
option(USE_CUDA "Build with CUDA support" ON)
option(BUILD_TESTS "Build unit tests" ON)

# Find CUDA only if requested
if(USE_CUDA)
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_ARCHITECTURES "89")
        message(STATUS "CUDA found and enabled")
    else()
        message(WARNING "CUDA requested but not found. Building CPU-only version.")
        set(USE_CUDA OFF)
    endif()
endif()

# Find pybind11
find_package(pybind11 REQUIRED)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# ============================================================================
# Python Module Target
# ============================================================================

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
    
    # Python module for linkage
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

# ============================================================================
# Test Target
# ============================================================================

if(BUILD_TESTS)
    # Find Google Test
    find_package(GTest)
    
    if(NOT GTest_FOUND)
        message(STATUS "Google Test not found. Downloading via FetchContent...")
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0
        )
        # For Windows: Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
    
    enable_testing()
    
    # Test executable
    if(USE_CUDA)
        # Tests with CUDA support
        add_executable(test_classifier 
            tests/test_classifier.cpp
            ${CPP_SOURCES}
            ${CUDA_SOURCES}
        )
        
        target_link_libraries(test_classifier 
            GTest::gtest 
            GTest::gtest_main
            CUDA::cudart
            CUDA::cuda_driver
        )
        
        set_target_properties(test_classifier PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
    else()
        # Tests without CUDA
        add_executable(test_classifier 
            tests/test_classifier.cpp
            ${CPP_SOURCES}
        )
        
        target_link_libraries(test_classifier 
            GTest::gtest 
            GTest::gtest_main
        )
    endif()
    
    target_include_directories(test_classifier PRIVATE ${PROJECT_SOURCE_DIR}/include)
    
    # Add test to CTest
    include(GoogleTest)
    gtest_discover_tests(test_classifier)
    
    message(STATUS "Tests will be built. Run with: make test or ctest")
endif()

# ============================================================================
# Build info
# ============================================================================

message(STATUS "=== Build Configuration ===")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "USE_CUDA: ${USE_CUDA}")
message(STATUS "BUILD_TESTS: ${BUILD_TESTS}")
message(STATUS "===========================")
