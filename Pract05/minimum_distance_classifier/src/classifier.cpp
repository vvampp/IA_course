// temporal filepath to avoid erorrs on development
#include "../include/classifier.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

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


void MinimumDistanceClassifier::fit(
  const std::vector<std::vector<float>>& X,
  const std::vector<int>& y)
{
  validate_data(X, y, true);

  n_features_ = static_cast<int>(X[0].size());
  n_classes_ = get_max_class(y) + 1;

  compute_centroids(X,y);

  if(use_cuda_ && cuda_available_){
    allocate_cuda_memory();
    transfer_centroids_to_device();
  }

}

void MinimumDistanceClassifier::validate_data(
  const std::vector<std::vector<float>>& X,
  const std::vector<int>& y,
  bool check_labels) const
{
  if (X.empty()){
    throw std::invalid_argument("Empty X... X cannot be emtpy");
  }

  // validate same ammount of features
  size_t n_features = X[0].size();
  if(n_features == 0){
    throw std::invalid_argument("Empty features... Features cannot be empty");
  }
  for(size_t i = 1; i < X.size(); ++i){
    if(X[i].size() != n_features){
      std::ostringstream oss;
      oss << "Inconsistent amount of features. Sample 0 has "
        << n_features << " features, but sample " << i 
        << " has " << X[i].size() << " features.";
      throw std::invalid_argument(oss.str());
    }
  }

  // verify compatibiility if already trained
  if(is_fitted_ && static_cast<int>(n_features) != n_features_){
      std::ostringstream oss;
      oss << "X has " << n_features << " features, but model was trained with "
      << n_features_ << " features.";
      throw std::invalid_argument(oss.str());
  }

  // verify tags if required
  if(check_labels){
    if(y.empty()){
      throw std::invalid_argument("y is empty... y cannot be empty");
    }

    if(X.size() != y.size()){
      std::ostringstream oss;
      oss << "X and y must have the same number of samples."
        << "X has " << X.size() << " samples, and y has " << y.size(); 
      throw std::invalid_argument(oss.str());
    }

    // check for negative tags
    for(size_t i = 0; i < y.size(); ++i){
      if(y[i] < 0){
        std::ostringstream oss;
        oss << "Label at index " << i << "is negative (" << y[i]
          << "). Labels must be all non-negative integers";
        throw std::invalid_argument(oss.str());
      }
    }
  }

  // validate missing features on X
  for(size_t i  = 0; i < X.size(); ++i){
    for(size_t j = 0; j < X[i].size(); ++j){
      if(std::isnan(X[i][j]) || std::isinf(X[i][j])){
        std::ostringstream oss;
        oss << "X contanins NaN of Inf at position (" << i << ", " << j << ")";
        throw std::invalid_argument(oss.str());
      }
    }
  }
}

void MinimumDistanceClassifier::compute_centroids(
  const std::vector<std::vector<float>>&X,
  const std::vector<int>& y)
{
  // sum accumulator vector
  std::vector<std::vector<float>> sums(n_classes_,
                                       std::vector<float>(n_features_, 0.0f));
  std::vector<int> counts(n_classes_, 0);


  for(size_t i = 0; i < X.size(); ++i){
    int label = y[i];
    // count amount of labels for each class on X
    counts[label]++;

    for(int j = 0; j < n_features_; ++j){
      // accumulate value of each feature on its respective class
      sums[label][j] += X[i][j];
    }
  }

  // calculate centroids
  centroids_.resize(n_classes_, std::vector<float>(n_features_, 0.0f));
  for(int c = 0; c < n_classes_; ++c){
    if(counts[c] == 0){
      std::cerr << "Warning: Class " << c << " has no samples."
        << "Centroid will be zero vector for " << c << "\n";
      continue;
    }

    for(int j = 0; j < n_features_; ++j){
      centroids_[c][j] = sums[c][j] / counts[c];
    }
  }

}

int MinimumDistanceClassifier::get_max_class(const std::vector<int>& y) const{
  if(y.empty()){
    return 0;
  }
  return *std::max_element(y.begin(), y.end());
}


}
