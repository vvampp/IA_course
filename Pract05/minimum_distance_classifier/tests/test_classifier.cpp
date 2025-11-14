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

class MinimumDistanceClassifierTest : public ::testing::Test {
  protected:
    void SetUp() override {
        X_simple = {{0.0f, 0.0f}, {0.1f, 0.1f}, {5.0f, 5.0f}, {5.1f, 5.1f}};
        y_simple = {0, 0, 1, 1};
    }

    std::vector<std::vector<float>> X_simple;
    std::vector<int> y_simple;
};

// Constructor Tests

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

// Basic testing

TEST_F(MinimumDistanceClassifierTest, FitBasic) {
    MinimumDistanceClassifier clf(false);

    EXPECT_NO_THROW(clf.fit(X_simple, y_simple));
    EXPECT_TRUE(clf.is_fitted());
    EXPECT_EQ(clf.get_n_classes(), 2);
    EXPECT_EQ(clf.get_n_features(), 2);
}

TEST_F(MinimumDistanceClassifierTest, PredictBasic) {
    MinimumDistanceClassifier clf(false);
    clf.fit(X_simple, y_simple);

    std::vector<std::vector<float>> X_test = {{0.0f, 0.0f}, {5.0f, 5.0f}};
    std::vector<int> predictions = clf.predict(X_test);

    EXPECT_EQ(predictions.size(), 2);
    EXPECT_EQ(predictions[0], 0);
    EXPECT_EQ(predictions[1], 1);
}

TEST_F(MinimumDistanceClassifierTest, GetBasicCentroids) {
    MinimumDistanceClassifier clf(false);
    clf.fit(X_simple, y_simple);

    auto centroids = clf.get_centroids();

    EXPECT_EQ(centroids.size(), 2);
    EXPECT_EQ(centroids[0].size(), 2);

    // check centroid for class 0
    EXPECT_NEAR(centroids[0][0], 0.05f, 1e-5);
    EXPECT_NEAR(centroids[0][1], 0.05f, 1e-5);

    // check centroids for class 1
    EXPECT_NEAR(centroids[1][0], 5.05f, 1e-5);
    EXPECT_NEAR(centroids[1][1], 5.05f, 1e-5);
}

TEST_F(MinimumDistanceClassifierTest, MoveAssingment) {
    MinimumDistanceClassifier clf1(false);
    clf1.fit(X_simple, y_simple);

    MinimumDistanceClassifier clf2(false);
    clf2 = std::move(clf1);

    EXPECT_TRUE(clf2.is_fitted());
    EXPECT_FALSE(clf1.is_fitted());
}

TEST_F(MinimumDistanceClassifierTest, MoveConstructor) {
    MinimumDistanceClassifier clf1(false);
    clf1.fit(X_simple, y_simple);

    MinimumDistanceClassifier clf2(std::move(clf1));

    EXPECT_TRUE(clf2.is_fitted());
    EXPECT_FALSE(clf1.is_fitted());
}

// main

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
