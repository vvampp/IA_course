#ifndef USE_CUDA

// temporal filepath for development
#include "../include/cuda_kernels.cuh"
#include <cfloat>
#include <vector>
#include <algorithm>
#include <string>

namespace mdc{
namespace cuda{

__global__ void compute_distances_kernel(
    const float* samples,
    const float* centroids,
    float* distances,
    int n_samples,
    int n_features,
    int n_classes)
{
  // (sample_idx, class_idx)
  int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int class_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if(sample_idx >= n_samples || class_idx >= n_classes){
    return;
  }

  float dist_sq = 0.0f;

  const float* sample = samples + sample_idx * n_features;
  const float* centroid = centroids + class_idx * n_features;

  // acumulate square diferences
  #pragma unroll 8
  for(int f = 0; f < n_features; ++f){
    float diff = sample[f] - centroid[f];
    dist_sq = fmaf(diff, diff, dist_sq); // Fused multiply - add
  }

  distances[sample_idx * n_classes + class_idx] = dist_sq;
}


__global__ void compute_distances_kernel_shared(
    const float* samples,
    const float* centroids,
    float* distances,
    int n_samples,
    int n_features,
    int n_classes)
{
  // shared mem to chache centroids
  extern __shared__ float shared_centroids[];

  int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int class_idx = blockIdx.y * blockDim.y + threadIdx.y;

  // load centroids into shared mem cooperatively
  // each thread loads one element
  int total_centroid_elements = n_features * n_classes;
  int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
  int threads_per_block = blockDim.x * blockDim.y;

  for(int i = thread_id; i < total_centroid_elements; i+=threads_per_block){
    shared_centroids[i] = centroids[i];
  }

  __syncthreads();

  if(sample_idx >= n_samples || class_idx >= n_classes){
    return;
  }

  // calculate distance with shared memory
  float dist_sq = 0.0f;

  const float* sample = samples + sample_idx * n_features;
  const float* centroid = shared_centroids + class_idx * n_features;

  #pragma unroll 8
  for(int f = 0; f < n_features; ++f){
    float diff = sample[f] - centroid[f];
    dist_sq = fmaf(diff,diff,dist_sq);
  }

  distances[sample_idx * n_classes + class_idx] = dist_sq;
}


__global__ void find_minimum_kernel(
    const float* distances,
    int* predictions,
    int n_samples,
    int n_classes)
{
  int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(sample_idx >= n_samples){
    return;
  }

  float min_dist = FLT_MAX;
  int best_class = 0;

  const float* sample_distances = distances + sample_idx * n_classes;

  for(int c = 0; c < n_classes; ++c){
    float dist = sample_distances[c];
    if(dist < min_dist){
      min_dist = dist;
      best_class = c;
    }
  }

  predictions[sample_idx] = best_class;
}


__global__ void find_minimum_kernel_parallell(
    const float* distances,
    int* predictions,
    int n_samples,
    int n_classes)
{
  int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(sample_idx >= n_samples){
    return;
  }

  const float* sample_distances = distances + sample_idx * n_classes;

  // each thread procecess multiple elements using reduction
  float min_dist = FLT_MAX;
  int best_class = 0;

  // first pass: find local minima
  for(int c = 0; c < n_classes; c += blockDim.y){
    float dist = sample_distances[c];
    if(dist < min_dist){
      min_dist = dist;
      best_class = c;
    }
  }

  // warp-level reduction using shuffle
  #pragma unroll
  for(int offset = 16; offset > 0; offset >>=1){
    float other_dist = __shfl_down_sync(0xffffffff, min_dist, offset);
    int other_class = __shfl_down_sync(0xffffffff, best_class, offset);

    if (other_dist < min_dist){
      min_dist = other_dist;
      best_class = other_class;
    }
  }
  // first thread and each warp writes the result
  if(threadIdx.y == 0){
    predictions[sample_idx] = best_class;
  }
}


// ditances in registries, minima without writting distances, prediction write on one function
__global__ void classify_fused_kernel(
    const float* samples,
    const float* centroids,
    int* predictions, 
    int n_samples,
    int n_features,
    int n_classes)
{
  int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(sample_idx >= n_samples){
    return;
  }

  const float* sample = samples + sample_idx * n_features;

  float min_dist = FLT_MAX;
  int best_class = 0;

  // calculate distance to each centroid and keep minima
  for(int c = 0; c < n_classes; ++c){
    const float* centroid = centroids + c * n_features;
    float dist_sq = 0.0f;
    #pragma unroll 8
    for(int f = 0; f < n_features; ++f){
      float diff = sample[f] - centroid[f];
      dist_sq = fmaf(diff, diff, dist_sq);
    }

    if(dist_sq < min_dist){
      min_dist = dist_sq;
      best_class = c;
    }
  }

  predictions[sample_idx] = best_class;
  
}


// host functions

bool check_cuda_available(){
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);

  if(error != cudaSuccess || device_count ==0){
    return false;
  }

  for(int i = 0 ; i < device_count; ++i){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    if(prop.major >= 3 && prop.minor >= 5){
      return true;
    }
  }

  return false;
}

std::string get_cuda_device_info(){
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    std::string info = "CUDA Devices: " + std::to_string(device_count) + "\n";
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        info += "Device " + std::to_string(i) + ": " + prop.name + "\n";
        info += "  Compute Capability: " + std::to_string(prop.major) + "." + 
                std::to_string(prop.minor) + "\n";
        info += "  Total Memory: " + 
                std::to_string(prop.totalGlobalMem / (1024*1024)) + " MB\n";
        info += "  Multiprocessors: " + std::to_string(prop.multiProcessorCount) + "\n";
        info += "  Max Threads per Block: " + 
                std::to_string(prop.maxThreadsPerBlock) + "\n";
    }
    
    return info;
}
  
  
 KernelConfig get_optimal_kernel_config(int n_samples, int n_classes, int n_features) {
    KernelConfig config;
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Use shared memory?
    size_t centroids_size = n_classes * n_features * sizeof(float);
    bool use_shared = (centroids_size <= MAX_SHARED_MEMORY_BYTES);
    
    if (use_shared) {
        config.shared_memory_size = centroids_size;
    } else {
        config.shared_memory_size = 0;
    }
    
    // configure dimentions
    int block_x = std::min(16, prop.maxThreadsDim[0]);
    int block_y = std::min(16, prop.maxThreadsDim[1]);
    
    // adjust if n_classes is small
    if (n_classes < block_y) {
        block_y = n_classes;
    }
    
    config.block_dim = dim3(block_x, block_y, 1);
    
    // Grid dim config
    int grid_x = (n_samples + block_x - 1) / block_x;
    int grid_y = (n_classes + block_y - 1) / block_y;
    
    config.grid_dim = dim3(grid_x, grid_y, 1);
    
    return config;
} 
  
}
}


#endif
