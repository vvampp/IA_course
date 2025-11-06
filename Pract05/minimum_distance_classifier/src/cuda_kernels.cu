#ifdef USE_CUDA

// temporal filepath for development
#include "cuda_kernels.cuh"
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
  #pragma unroll
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

    if(prop.major >= 3 || (prop.major == 3 && prop.minor >= 5)){
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


void cuda_classify(
    const float* h_samples,
    const float* d_centroids,
    int* h_predictions,
    int n_samples,
    int n_features,
    int n_classes)
{
  // allocate device memory
  float* d_samples = nullptr;
  float* d_distances = nullptr;
  int* d_predictions = nullptr;

  size_t samples_size = n_samples * n_features * sizeof(float);
  size_t distance_size = n_samples * n_classes * sizeof(float);
  size_t predictions_size = n_samples * sizeof(int);

  CUDA_CHECK(cudaMalloc(&d_samples, samples_size));
  CUDA_CHECK(cudaMalloc(&d_distances, distance_size));
  CUDA_CHECK(cudaMalloc(&d_predictions, predictions_size));

  CUDA_CHECK(cudaMemcpy(d_samples, h_samples, samples_size, cudaMemcpyHostToDevice));

  // get optimal configuration for the kernel given the function arguments
  KernelConfig config = get_optimal_kernel_config(n_samples, n_classes, n_features);

  size_t centroids_size = n_classes * n_features * sizeof(float);

  if(centroids_size <= MAX_SHARED_MEMORY_BYTES && n_classes <= MAX_CLASSES_SHARED){
    // use kernel with shared memory
    compute_distances_kernel_shared<<<config.grid_dim, config.block_dim, config.shared_memory_size>>>(
        d_samples, d_centroids, d_distances,
        n_samples, n_features, n_classes
        );
  } else if (n_samples > 10000 && n_classes < 20){
    // limited amount of classes, fused kernel
    int threads_per_block = 256;
    int blocks = (n_samples + threads_per_block - 1) / threads_per_block;

    classify_fused_kernel<<<blocks, threads_per_block>>>(
        d_samples, d_centroids, d_predictions,
        n_samples, n_features, n_classes
        );

    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_predictions, d_predictions, predictions_size, cudaMemcpyDeviceToHost));

    cudaFree(d_samples);
    cudaFree(d_distances);
    cudaFree(d_predictions);
    return;

  } else {
    // standard kernel without shared memory
    compute_distances_kernel<<<config.grid_dim, config.block_dim>>>(
        d_samples, d_centroids, d_distances,
        n_samples, n_features, n_classes
        );
  }

  CUDA_KERNEL_CHECK();

  // launch minimin search kernel
  int threads_per_block = std::min(256,n_samples);
  int blocks = (n_samples + threads_per_block -1)/ threads_per_block;

  if(n_classes > 64){
    // parallel reduction if too much classes
    dim3 block_dim(threads_per_block, 8, 1);
    find_minimum_kernel_parallell<<<blocks,block_dim>>>(
        d_distances, d_predictions, n_samples, n_classes
        );
  } else {
    // secuential reduction for small amout of classes
    find_minimum_kernel<<<blocks,threads_per_block>>>(
        d_distances, d_predictions, n_samples, n_classes
        );
  }

  CUDA_KERNEL_CHECK();


  CUDA_CHECK(cudaMemcpy(h_predictions, d_predictions, predictions_size, cudaMemcpyDeviceToHost));

  cudaFree(d_samples);
  cudaFree(d_predictions);
  cudaFree(d_distances);
}


void cuda_classify_streams(
    const float* h_samples,
    const float* d_centroids,
    int* h_predictions,
    int n_samples,
    int n_features,
    int n_classes,
    int n_streams)
{
  int chunk_size = (n_samples + n_streams - 1) / n_streams;

  std::vector<cudaStream_t> streams(n_streams);
  for(int i = 0; i < n_streams; ++i){
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
  }

  // allocte pinned memory
  // useful for async transfers
  float* h_samples_pinned;
  int* h_predictions_pinned;
  CUDA_CHECK(cudaMallocHost(&h_samples_pinned, n_samples * n_features * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_predictions_pinned, n_samples * sizeof(int)));

  // copy to pinned memory
  std::copy(h_samples, h_samples + n_samples * n_features, h_samples_pinned);

  // parallel chunk processing
  for(int i = 0 ; i < n_streams; ++i){
    int offset = i * chunk_size;
    int current_chunk_size = std::min(chunk_size, n_samples - offset);
    
    if(current_chunk_size <= 0){
      break;
    }

    float* d_samples_chunk;
    int* d_predictions_chunk;

    size_t samples_chunk_size = current_chunk_size * n_features * sizeof(float);
    size_t predictions_chunk_size = current_chunk_size * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_samples_chunk,samples_chunk_size));
    CUDA_CHECK(cudaMalloc(&d_predictions_chunk, predictions_chunk_size));

    // async transfer H2D
    cuda_memcpy_async_htod(
        d_samples_chunk,
        h_samples_pinned + offset * n_features,
        samples_chunk_size,
        streams[i]
        );

    int threads = 256;
    int blocks = (current_chunk_size + threads - 1) / threads;

    classify_fused_kernel<<<blocks, threads, 0, streams[i]>>>(
        d_samples_chunk, d_centroids, d_predictions_chunk,
        current_chunk_size, n_features, n_classes
        );

    // async transfer D2H
    cuda_memcpy_async_dtoh_int(
        h_predictions_pinned + offset,
        d_predictions_chunk,
        predictions_chunk_size,
        streams[i]
        );
    // free mem after sync
  }

  // sync streams
  for(int i = 0; i < n_streams; ++i){
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
  }

  // copy final predictions
  std::copy(h_predictions_pinned,
      h_predictions_pinned + n_samples,
      h_predictions);

  cudaFreeHost(h_samples_pinned);
  cudaFreeHost(h_predictions_pinned);
}

// memory management functions

void cuda_malloc(float** d_ptr, size_t size){
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(d_ptr), size));
}

void cuda_free(void* d_ptr){
  if(d_ptr != nullptr){
    CUDA_CHECK(cudaFree(d_ptr));
  }
}

void cuda_memcpy_host_to_device(float* d_dst,
    const float* h_src,
    size_t size)
{
  CUDA_CHECK(cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice));
}

void cuda_memcpy_device_to_host(float* h_dst,
    const float* d_src,
    size_t size)
{
  CUDA_CHECK(cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost));
}

void cuda_memcpy_async_htod(float* d_dst,
    const float* h_src,
    size_t size,
    cudaStream_t stream)
{
  CUDA_CHECK(cudaMemcpyAsync(d_dst, h_src, size, cudaMemcpyHostToDevice, stream));
}

void cuda_memcpy_async_htod_int(int* d_dst,
    const int* h_src,
    size_t size, 
   cudaStream_t stream)
{
    CUDA_CHECK(cudaMemcpyAsync(d_dst, h_src, size, cudaMemcpyHostToDevice, stream));
}

void cuda_memcpy_async_dtoh_int(int* h_dst,
    const int* d_src,
    size_t size,
    cudaStream_t stream)
{
    CUDA_CHECK(cudaMemcpyAsync(h_dst, d_src, size, cudaMemcpyDeviceToHost, stream));
}

void cuda_memcpy_async_dtoh(float* h_dst,
    const float* d_src,
    size_t size,
    cudaStream_t stream)
{
  CUDA_CHECK(cudaMemcpyAsync(h_dst, d_src, size, cudaMemcpyDeviceToHost, stream));
}

  
}
}


#endif
