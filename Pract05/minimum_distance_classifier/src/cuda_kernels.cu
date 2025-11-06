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

}
}


#endif
