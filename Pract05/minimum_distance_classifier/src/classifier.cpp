// temporal filepath to avoid erorrs on development
#include "../include/classifier.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
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

  is_fitted_ = true;

}


std::vector<int> MinimumDistanceClassifier::predict(
  const std::vector<std::vector<float>>& X) const
{
  if (!is_fitted_){
    throw std::runtime_error("Model not fitted. Call fit() before predict()");
  }

  validate_data(X);
  if (X.empty()){
    return std::vector<int>();
  }

  if (use_cuda_ && cuda_available_){
    return predict_cuda(X);
  } else {
    return predict_cpu(X);
  }
}


std::vector<int> MinimumDistanceClassifier::predict_batch(
  const std::vector<std::vector<float>>& X) const
{
  return predict(X);
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


float MinimumDistanceClassifier::euclidean_distance_squared(
  const std::vector<float>& sample,
  const std::vector<float>& centroid) const
{
  float dist_sq = 0.0f;
  for(size_t i = 0; i < sample.size(); ++i){
    float diff = sample[i] - centroid[i];
    dist_sq += diff * diff;
  }
  return dist_sq;
}


std::vector<int> MinimumDistanceClassifier::predict_cpu(
  const std::vector<std::vector<float>>& X) const
{
  std::vector<int> predictions(X.size());

  for(size_t i = 0; i < X.size(); ++i){
    float min_distance = std::numeric_limits<float>::max();
    int best_class = 0;

    for(int c = 0; c < n_classes_ ; ++c){
      float dist = euclidean_distance_squared(X[i], centroids_[c]);
      if(dist < min_distance){
        min_distance = dist;
        best_class = c;
      }
    }
    predictions[i] = best_class;
  }

  return predictions;
}

std::vector<int> MinimumDistanceClassifier::predict_cuda(
  const std::vector<std::vector<float>>& X) const
{
#ifdef USE_CUDA
  // X to contigous
  int n_samples = static_cast<int>(X.size());
  std::vector<float> X_flat(n_samples * n_features);

  for(int i = 0; i < n_samples; ++i){
    for(int j = 0; j < n_features_; ++j){
      X_flat[i * n_features_ + j] = X[i][j];
    }
  }

  // call to CUDA wrapper (pending implementation)
  std::vector<int> predictions(n_samples);

  cuda_classify(
    X_flat.data(),
    d_centroids_,
    predictions.data(),
    n_samples,
    n_features_,
    n_classes_
  );

  return predictions;
#else
  // fallback
  return predict_cpu(X);
#endif
}

int MinimumDistanceClassifier::get_max_class(const std::vector<int>& y) const{
  if(y.empty()){
    return 0;
  }
  return *std::max_element(y.begin(), y.end());
}


bool MinimumDistanceClassifier::initialize_cuda(){
#ifdef USE_CUDA 
  return check_cuda_available();
#else
  return false;
#endif
}


void MinimumDistanceClassifier::allocate_cuda_memory(){
#ifdef USE_CUDA
  free_cuda_memory();
  centroids_size_ = n_classes_ * n_features_ * sizeof(float);
  cuda_malloc(&d_centroids_, centroids_size_);
#endif
}


void free_cuda_memory(){
#ifdef USE_CUDA
  if(d_centroids_ != nullptr){
    cuda_free(d_centroids_);
    d_centroids_ = nullptr;
    centroids_size_ = 0;
  }
#endif
}


void transfer_centroids_to_device(){
#ifdef USE_CUDA
  std::vector<float> centroids_flat(n_classes_ * n_features_);
  for(int c = 0; c < n_classes_; ++c){
    for(int f = 0; f < n_features_; ++f ){
      centroids_flat[c * n_features + f] = centroids_[c][f];
    }
  }
  cuda_memcpy_hots_to_device(d_centroids_, centroids_flat.data(), centroids_size_)
#endif
}


}
