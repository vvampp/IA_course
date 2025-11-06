#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

#include <vector>
namespace mdc {

class MinimumDistanceClassifier {

public:

  explicit MinimumDistanceClassifier(bool use_cuda = false);
  ~MinimumDistanceClassifier();

  // prohibit copies ... CUDA good practice regarding pointers
  MinimumDistanceClassifier(const MinimumDistanceClassifier&) = delete;
  MinimumDistanceClassifier& operator=(const MinimumDistanceClassifier&& other) noexcept;

  void fit(const std::vector<std::vector<float>>& X, const std::vector<int> y);
  std::vector<int> predict(const std::vector<std::vector<float>>& X) const;
  std::vector<int> predict_batch(const std::vector<std::vector<float>>& X) const;

  std::vector<std::vector<float>> get_centroids() const;
  
  // get member attributes
  int get_n_classes() const { return n_classes_; }
  int get_n_features() const { return n_features_; }

  // check for status
  bool is_using_cuda() const { return use_cuda_ && cuda_available_; }
  bool is_fitted() const { return is_fitted_; }

private:

  std::vector<std::vector<float>> centroids_;
  int n_classes_;
  int n_features_;
  int use_cuda_;
  int cuda_available_;

  float* d_centroids_; 
  size_t centroids_size_;


};

}

#endif
