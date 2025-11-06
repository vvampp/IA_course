#include <__clang_cuda_builtin_vars.h>
#ifndef USE_CUDA

// temporal filepath for development
#include "../include/cuda_kernels.cuh"
#include <cfloat>
#include <vector>
#include <algorithm>

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
  const float* centroid = centroids + class_idx * n_features;

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
  
  
  
  
}
}


#endif
