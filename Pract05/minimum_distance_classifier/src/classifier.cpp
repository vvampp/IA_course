// temporal filepath to avoid erorrs on development
#include "../include/classifier.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>

#ifdef USE_CUDA
#include "cuda_kernels.cuh"
#endif

namespace mdc {

MinimumDistanceClassifier::MinimumDistanceClassifier(bool use_cuda)
  : n_classes_(0)
  , n_features_(0)
  , is_fitted_(false)
  , use_cuda_(use_cuda)
  , d_centroids_(nullptr)
  , centroids_size_(0)
{
  #ifdef USE_CUDA
    if(use_cuda_){
      cuda_availabe_ = initialize_cuda();
      if(!cuda_availabe_){
        std::cer << "CUDA requested but not available. Falling back to CPU.\n";
      }
    }
  #else
    if(use_cuda_){
      std::cerr << "Compiled without CUDA support. Using CPU only.\n";
      use_cuda_ = false;
    }
  #endif
}

MinimumDistanceClassifier::~MinimumDistanceClassifier(){
  free_cuda_memory();
}


}

