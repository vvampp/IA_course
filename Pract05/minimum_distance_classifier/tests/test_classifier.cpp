#include "classifier.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace mdc;

// helper functions
std::vector<std::vector<float>> generate_2d_clusters(int n_samples_per_class, int n_classes,
                                                     int n_features, float separation = 5.0f,
                                                     unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<std::vector<float>> X;
    X.reserve(n_samples_per_class * n_classes);

    for (int c = 0; c < n_classes; ++c) {
        std::vector<float> center(n_features);
        for (int f = 0; f < n_features; ++f) {
            center[f] = c * separation;
        }

        for (int i = 0; i < n_samples_per_class; ++i) {
            std::vector<float> sample(n_features);
            for (int f = 0; f < n_features; ++f) {
                sample[f] = center[f] + dist(rng);
            }
            X.push_back(sample);
        }
    }
    return X;
}

std::vector<int> generate_labels(int n_samples_per_class, int n_classes) {
    std::vector<int> y;
    y.reserve(n_samples_per_class * n_classes);

    for (int c = 0; c < n_classes; ++c) {
        for (int i = 0; i < n_samples_per_class; ++i) {
            y.push_back(c);
        }
    }
    return y;
}

float calculate_accuracy(const std::vector<int> &y_true, const std::vector<int> &y_pred) {
    if (y_true.size() != y_pred.size())
        return 0.0f;

    int correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i)
        if (y_true[i] == y_pred[i])
            correct += 1;

    return static_cast<float>(correct) / y_true.size();
}

// Constructor Tests

class MinimumDistanceClassifierTest : public ::testing::Test {
  protected:
    void SetUp() override {
        X_simple = {{0.0f, 0.0f}, {0.1f, 0.1f}, {5.0f, 5.0f}, {5.1f, 5.1f}};
        y_simple = {0, 0, 1, 1};
    }

    std::vector<std::vector<float>> X_simple;
    std::vector<int> y_simple;
};

TEST_F(MinimumDistanceClassifierTest, ConstructorCPU) {
    EXPECT_NO_THROW(MinimumDistanceClassifier clf(false));

    MinimumDistanceClassifier clf(false);
    EXPECT_FALSE(clf.is_using_cuda());
    EXPECT_FALSE(clf.is_fitted());
}

TEST_F(MinimumDistanceClassifierTest, ConstructorCUDA) {
    MinimumDistanceClassifier clf(true);

    EXPECT_FALSE(clf.is_fitted());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
