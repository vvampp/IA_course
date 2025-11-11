#pragma once

#include <stdexcept>
#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <string>

namespace mdc {
namespace cuda {

constexpr int MAX_THREADS_PER_BLOCK = 256;
constexpr int MAX_SHARED_MEMORY_BYTES = 48 * 1024; // 48KB
constexpr int MAX_FEATURES_SHARED = 512;
constexpr int MAX_CLASSES_SHARED = 128;

// error handling macros

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t error = call;                                                                  \
        if (error != cudaSuccess) {                                                                \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "                  \
                      << cudaGetErrorString(error) << std::endl;                                   \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error));     \
        }                                                                                          \
    } while (0)

#define CUDA_KERNEL_CHECK()                                                                        \
    do {                                                                                           \
        cudaError_t error = cudaGetLastError();                                                    \
        if (error != cudaSuccess) {                                                                \
            std::cerr << "CUDA kernel launch error at " << __FILE__ << ":" << __LINE__ << " - "    \
                      << cudaGetErrorString(error) << std::endl;                                   \
        }                                                                                          \
        CUDA_CHECK(cudaDeviceSynchronize());                                                       \
    } while (0)

struct KernelConfig {
    dim3 grid_dim;
    dim3 block_dim;
    size_t shared_memory_size;
};

__global__ void compute_distances_kernel(const float *samples, const float *centroids,
                                         float *distances, int n_samples, int n_features,
                                         int n_classes);

__global__ void compute_distances_kernel_shared(const float *samples, const float *centroids,
                                                float *distances, int n_samples, int n_features,
                                                int n_classes);

__global__ void find_minimum_kernel(const float *distances, int *predictions, int n_samples,
                                    int n_classes);

__global__ void find_minimum_kernel_parallel(const float *distances, int *predictions,
                                             int n_samples, int n_classes);

__global__ void classify_fused_kernel(const float *samples, const float *centroids,
                                      int *predictions, int n_samples, int n_features,
                                      int n_classes);

// wrappers
bool check_cuda_available();

std::string get_cuda_device_info();

KernelConfig get_optimal_kernel_config(int n_samples, int n_clases, int n_features);

void cuda_classify(const float *h_samples, const float *d_centroids, int *h_predictions,
                   int n_samples, int n_features, int n_classes);

void cuda_classify_streams(const float *h_samples, const float *d_centroids, int *h_predictions,
                           int n_samples, int n_features, int n_classes, int n_streams = 4);

// memory management
void cuda_malloc(float **d_ptr, size_t size);

void cuda_free(void *d_ptr);

void cuda_memcpy_host_to_device(float *d_dst, const float *h_src, size_t size);

void cuda_memcpy_device_to_host(float *h_dst, const float *d_src, size_t size);

void cuda_memcpy_async_htod(float *d_dst, const float *h_src, size_t size, cudaStream_t stream);

void cuda_memcpy_async_htod_int(int *d_dst, const int *h_src, size_t size, cudaStream_t stream);

void cuda_memcpy_async_dtoh_int(int *h_dst, const int *d_src, size_t size, cudaStream_t stream);

void cuda_memcpy_async_dtoh(float *h_dst, const float *d_src, size_t size, cudaStream_t stream);

} // namespace cuda
} // namespace mdc

#endif
