#include "classifier.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <random>
#include <set>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace mdc;

// DATASET STRUCTURE AND GENERATION

struct Dataset {
    std::vector<std::vector<float>> X;
    std::vector<int> y;
};

Dataset generate_realistic_dataset(int n_samples_per_class,
                                    int n_classes,
                                    int n_features,
                                    float density = 1.0f,       // How packed the space is (0.5-2.0)
                                    float outlier_rate = 0.05f, // Percentage of outliers (0.0-0.2)
                                    unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);
    
    Dataset dataset;
    int total_samples = n_samples_per_class * n_classes;
    dataset.X.reserve(total_samples);
    dataset.y.reserve(total_samples);
    
    // Generate cluster centers using natural placement strategies
    std::vector<std::vector<float>> centroids(n_classes);
    
    // Strategy: Some clusters form groups (natural clustering of clusters)
    int n_groups = std::max(1, n_classes / 4);  // Clusters form ~4 groups
    std::vector<std::vector<float>> group_centers(n_groups);
    
    // Place group centers far apart
    for (int g = 0; g < n_groups; ++g) {
        group_centers[g].resize(n_features);
        for (int f = 0; f < n_features; ++f) {
            group_centers[g][f] = g * 50.0f / density + normal_dist(rng) * 5.0f;
        }
    }
    
    // Place individual clusters near their group centers
    for (int c = 0; c < n_classes; ++c) {
        int group = c % n_groups;
        centroids[c].resize(n_features);
        
        for (int f = 0; f < n_features; ++f) {
            // Offset from group center with some randomness
            float offset = (uniform_dist(rng) - 0.5f) * 20.0f / density;
            centroids[c][f] = group_centers[group][f] + offset;
        }
    }
    
    // Generate cluster-specific parameters (variable spread and size)
    std::vector<float> cluster_spreads(n_classes);
    std::vector<int> cluster_sizes(n_classes);
    
    for (int c = 0; c < n_classes; ++c) {
        // Variable spread: some clusters tight, some loose
        cluster_spreads[c] = 1.0f + uniform_dist(rng) * 3.0f;  // Range: 1.0 to 4.0
        
        // Variable size: some clusters bigger than others
        float size_multiplier = 0.5f + uniform_dist(rng) * 1.5f;  // Range: 0.5x to 2x
        cluster_sizes[c] = static_cast<int>(n_samples_per_class * size_multiplier);
    }
    
    // Normalize total samples
    int total_cluster_samples = std::accumulate(cluster_sizes.begin(), cluster_sizes.end(), 0);
    for (int c = 0; c < n_classes; ++c) {
        cluster_sizes[c] = static_cast<int>(
            (static_cast<float>(cluster_sizes[c]) / total_cluster_samples) * 
            n_samples_per_class * n_classes * (1.0f - outlier_rate)
        );
    }
    
    // Generate samples for each cluster
    for (int c = 0; c < n_classes; ++c) {
        std::normal_distribution<float> cluster_dist(0.0f, cluster_spreads[c]);
        
        // Decide if this cluster has substructure (20% chance)
        bool has_substructure = uniform_dist(rng) < 0.2f;
        std::vector<float> subcluster_offset(n_features, 0.0f);
        
        if (has_substructure) {
            for (int f = 0; f < n_features; ++f) {
                subcluster_offset[f] = (uniform_dist(rng) - 0.5f) * 8.0f;
            }
        }
        
        for (int i = 0; i < cluster_sizes[c]; ++i) {
            std::vector<float> sample(n_features);
            
            // 30% of samples go to subcluster if it exists
            bool use_subcluster = has_substructure && (uniform_dist(rng) < 0.3f);
            
            for (int f = 0; f < n_features; ++f) {
                float base = centroids[c][f];
                if (use_subcluster) {
                    base += subcluster_offset[f];
                }
                sample[f] = base + cluster_dist(rng);
            }
            
            dataset.X.push_back(sample);
            dataset.y.push_back(c);
        }
    }
    
    // Add outliers 
    int n_outliers = static_cast<int>(total_samples * outlier_rate);
    std::uniform_int_distribution<int> class_dist(0, n_classes - 1);
    
    for (int i = 0; i < n_outliers; ++i) {
        std::vector<float> outlier(n_features);
        
        // Outliers placed randomly in the space, far from clusters
        for (int f = 0; f < n_features; ++f) {
            outlier[f] = uniform_dist(rng) * 100.0f - 50.0f;
        }
        
        dataset.X.push_back(outlier);
        dataset.y.push_back(class_dist(rng));  // Random class label
    }
    
    // Shuffle to mix everything
    std::vector<size_t> indices(dataset.X.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    Dataset shuffled;
    shuffled.X.reserve(dataset.X.size());
    shuffled.y.reserve(dataset.y.size());
    
    for (const auto& idx : indices) {
        shuffled.X.push_back(dataset.X[idx]);
        shuffled.y.push_back(dataset.y[idx]);
    }
    
    return shuffled;
}

Dataset generate_easy_realistic_dataset(int n_samples_per_class,
                                         int n_classes,
                                         int n_features,
                                         unsigned seed = 42) {
    return generate_realistic_dataset(n_samples_per_class, n_classes, n_features,
                                      0.5f,   // Low density (more separation)
                                      0.02f,  // 2% outliers
                                      seed);
}

Dataset generate_medium_realistic_dataset(int n_samples_per_class,
                                           int n_classes,
                                           int n_features,
                                           unsigned seed = 42) {
    return generate_realistic_dataset(n_samples_per_class, n_classes, n_features,
                                      1.0f,   // Medium density
                                      0.05f,  // 5% outliers
                                      seed);
}

Dataset generate_hard_realistic_dataset(int n_samples_per_class,
                                         int n_classes,
                                         int n_features,
                                         unsigned seed = 42) {
    return generate_realistic_dataset(n_samples_per_class, n_classes, n_features,
                                      2.0f,   // High density (less separation)
                                      0.15f,  // 15% outliers
                                      seed);
}

// HELPER FUNCTIONS

float calculate_accuracy(const std::vector<int> &y_true, const std::vector<int> &y_pred) {
    if (y_true.size() != y_pred.size())
        return 0.0f;

    int correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i)
        if (y_true[i] == y_pred[i])
            correct += 1;

    return static_cast<float>(correct) / y_true.size();
}

// BASIC FUNCTIONALITY TESTS

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

    EXPECT_NEAR(centroids[0][0], 0.05f, 1e-5);
    EXPECT_NEAR(centroids[0][1], 0.05f, 1e-5);
    EXPECT_NEAR(centroids[1][0], 5.05f, 1e-5);
    EXPECT_NEAR(centroids[1][1], 5.05f, 1e-5);
}

TEST_F(MinimumDistanceClassifierTest, MoveConstructor) {
    MinimumDistanceClassifier clf1(false);
    clf1.fit(X_simple, y_simple);

    MinimumDistanceClassifier clf2(std::move(clf1));

    EXPECT_TRUE(clf2.is_fitted());
    EXPECT_FALSE(clf1.is_fitted());
}

TEST_F(MinimumDistanceClassifierTest, MoveAssingment) {
    MinimumDistanceClassifier clf1(false);
    clf1.fit(X_simple, y_simple);

    MinimumDistanceClassifier clf2(false);
    clf2 = std::move(clf1);

    EXPECT_TRUE(clf2.is_fitted());
    EXPECT_FALSE(clf1.is_fitted());
}

// VALIDATION TESTS

TEST_F(MinimumDistanceClassifierTest, EmtpyData) {
    MinimumDistanceClassifier clf(false);
    std::vector<std::vector<float>> X_empty;
    std::vector<int> y_empty;

    EXPECT_THROW(clf.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(MinimumDistanceClassifierTest, EmtpyTags) {
    MinimumDistanceClassifier clf(false);
    std::vector<std::vector<float>> X = {{1.0f, 2.0f}};
    std::vector<int> y_empty;

    EXPECT_THROW(clf.fit(X, y_empty), std::invalid_argument);
}

TEST_F(MinimumDistanceClassifierTest, EmtpyFeatures) {
    MinimumDistanceClassifier clf(false);
    std::vector<std::vector<float>> X = {{}};
    std::vector<int> y = {0};

    EXPECT_THROW(clf.fit(X, y), std::invalid_argument);
}

TEST_F(MinimumDistanceClassifierTest, MismatchedSizes) {
    MinimumDistanceClassifier clf(false);
    std::vector<std::vector<float>> X = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    std::vector<int> y = {0};

    EXPECT_THROW(clf.fit(X, y), std::invalid_argument);
}

TEST_F(MinimumDistanceClassifierTest, InconsistentFeatures) {
    MinimumDistanceClassifier clf(false);
    std::vector<std::vector<float>> X = {{1.0f, 2.0f}, {3.0f, 4.0f, 5.0f}};
    std::vector<int> y = {0, 1};

    EXPECT_THROW(clf.fit(X, y), std::invalid_argument);
}

TEST_F(MinimumDistanceClassifierTest, NegativeLabels) {
    MinimumDistanceClassifier clf(false);
    std::vector<std::vector<float>> X = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    std::vector<int> y = {0, -1};

    EXPECT_THROW(clf.fit(X, y), std::invalid_argument);
}

TEST_F(MinimumDistanceClassifierTest, NaNInData) {
    MinimumDistanceClassifier clf(false);
    std::vector<std::vector<float>> X = {{1.0f, 2.0f}, {NAN, 4.0f}};
    std::vector<int> y = {0, 1};

    EXPECT_THROW(clf.fit(X, y), std::invalid_argument);
}

TEST_F(MinimumDistanceClassifierTest, InfInData) {
    MinimumDistanceClassifier clf(false);
    std::vector<std::vector<float>> X = {{1.0f, 2.0f}, {INFINITY, 4.0f}};
    std::vector<int> y = {0, 1};

    EXPECT_THROW(clf.fit(X, y), std::invalid_argument);
}

TEST_F(MinimumDistanceClassifierTest, PredictBeforeFit) {
    MinimumDistanceClassifier clf(false);
    std::vector<std::vector<float>> X = {{1.0f, 2.0f}};

    EXPECT_THROW(clf.predict(X), std::runtime_error);
}

TEST_F(MinimumDistanceClassifierTest, GetCentroidsBeforeFit) {
    MinimumDistanceClassifier clf(false);

    EXPECT_THROW(clf.get_centroids(), std::runtime_error);
}

TEST_F(MinimumDistanceClassifierTest, WrongFeatureCountOnPredict) {
    MinimumDistanceClassifier clf(false);
    clf.fit(X_simple, y_simple);

    std::vector<std::vector<float>> X_wrong = {{1.0f, 2.0f, 3.0f}};

    EXPECT_THROW(clf.predict(X_wrong), std::invalid_argument);
}

TEST_F(MinimumDistanceClassifierTest, MaxClassesLimit) {
    MinimumDistanceClassifier clf(false);

    std::vector<std::vector<float>> X = {{1.0f, 2.0f}};
    std::vector<int> y = {100000};

    EXPECT_THROW(clf.fit(X, y), std::invalid_argument);
}

// ACCURACY TESTS 

TEST(ClassifierAccuracyTests, WellSeparatedClusters2D) {
    auto dataset = generate_easy_realistic_dataset(50, 3, 2, 100);
    
    MinimumDistanceClassifier clf(false);
    clf.fit(dataset.X, dataset.y);

    auto predictions = clf.predict(dataset.X);
    float accuracy = calculate_accuracy(dataset.y, predictions);

    EXPECT_GT(accuracy, 0.85f);  // Easy dataset should have high accuracy
}

TEST(ClassifierAccuracyTests, WellSeparatedClusters10D) {
    // Easy dataset: 4 classes, higher dimensional
    auto dataset = generate_easy_realistic_dataset(30, 4, 10, 200);
    
    MinimumDistanceClassifier clf(false);
    clf.fit(dataset.X, dataset.y);

    auto predictions = clf.predict(dataset.X);
    float accuracy = calculate_accuracy(dataset.y, predictions);

    EXPECT_GT(accuracy, 0.80f);  // Still easy but more dimensions
}

TEST(ClassifierAccuracyTests, OverlappingClusters) {
    // Hard dataset: overlapping clusters
    auto dataset = generate_hard_realistic_dataset(50, 2, 2, 300);
    
    MinimumDistanceClassifier clf(false);
    clf.fit(dataset.X, dataset.y);

    auto predictions = clf.predict(dataset.X);
    float accuracy = calculate_accuracy(dataset.y, predictions);

    EXPECT_GT(accuracy, 0.50f);  // Hard dataset, but better than random
}

TEST(ClassifierAccuracyTests, HeavyWorkload) {
    // Medium difficulty: many classes
    auto dataset = generate_medium_realistic_dataset(200, 50, 120, 400);
    
    MinimumDistanceClassifier clf(false);
    clf.fit(dataset.X, dataset.y);

    auto predictions = clf.predict(dataset.X);
    float accuracy = calculate_accuracy(dataset.y, predictions);

    EXPECT_GT(accuracy, 0.40f);  // 50 classes, medium difficulty
}

// EDGE CASES TESTS

TEST(ClassifierEdgeCases, SingleSample) {
    MinimumDistanceClassifier clf(false);
    std::vector<std::vector<float>> X = {{1.0f, 2.0f}};
    std::vector<int> y = {0};

    EXPECT_NO_THROW(clf.fit(X, y));

    auto predictions = clf.predict(X);
    EXPECT_EQ(predictions[0], 0);
}

TEST(ClassifierEdgeCases, SingleClass) {
    MinimumDistanceClassifier clf(false);
    std::vector<std::vector<float>> X = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
    std::vector<int> y = {0, 0, 0};

    EXPECT_NO_THROW(clf.fit(X, y));

    auto predictions = clf.predict(X);
    for (const auto &pred : predictions) {
        EXPECT_EQ(pred, 0);
    }
}

TEST(ClassifierEdgeCases, ManyClasses) {
    // Easy dataset with many classes
    auto dataset = generate_easy_realistic_dataset(5, 100, 10, 500);
    
    MinimumDistanceClassifier clf(false);
    EXPECT_NO_THROW(clf.fit(dataset.X, dataset.y));

    EXPECT_EQ(clf.get_n_classes(), 100);
}

TEST(ClassifierEdgeCases, HighDimensional) {
    // Medium dataset with high dimensions
    auto dataset = generate_medium_realistic_dataset(10, 3, 500, 600);
    
    MinimumDistanceClassifier clf(false);
    EXPECT_NO_THROW(clf.fit(dataset.X, dataset.y));

    EXPECT_EQ(clf.get_n_features(), 500);
}

TEST(ClassifierEdgeCases, LargeDataset) {
    auto dataset = generate_easy_realistic_dataset(1000, 3, 20, 700);
    
    MinimumDistanceClassifier clf(false);
    EXPECT_NO_THROW(clf.fit(dataset.X, dataset.y));

    auto predictions = clf.predict(dataset.X);
    EXPECT_EQ(predictions.size(), 3000);
}

TEST(ClassifierEdgeCases, EmptyPredictionSet) {
    auto dataset = generate_easy_realistic_dataset(10, 2, 5, 800);
    
    MinimumDistanceClassifier clf(false);
    clf.fit(dataset.X, dataset.y);

    std::vector<std::vector<float>> X_empty;
    EXPECT_THROW(clf.predict(X_empty), std::invalid_argument);
}

TEST(ClassifierEdgeCases, IdenticalSamples) {
    MinimumDistanceClassifier clf(false);
    std::vector<std::vector<float>> X = {{1.0f, 2.0f}, {1.0f, 2.0f}, {3.0f, 4.0f}, {3.0f, 4.0f}};
    std::vector<int> y = {0, 0, 1, 1};

    EXPECT_NO_THROW(clf.fit(X, y));

    auto predictions = clf.predict(X);
    float accuracy = calculate_accuracy(y, predictions);
    EXPECT_EQ(accuracy, 1.0f);
}

// DETERMINISM TESTS

TEST(ClassifierDeterminism, ConsistentResults) {
    auto dataset = generate_medium_realistic_dataset(50, 3, 5, 12345);
    
    MinimumDistanceClassifier clf1(false);
    MinimumDistanceClassifier clf2(false);

    clf1.fit(dataset.X, dataset.y);
    clf2.fit(dataset.X, dataset.y);

    auto pred1 = clf1.predict(dataset.X);
    auto pred2 = clf2.predict(dataset.X);

    EXPECT_EQ(pred1, pred2);
}

TEST(ClassifierDeterminism, ConsistentCentroids) {
    auto dataset = generate_medium_realistic_dataset(50, 3, 5, 54321);
    
    MinimumDistanceClassifier clf1(false);
    MinimumDistanceClassifier clf2(false);

    clf1.fit(dataset.X, dataset.y);
    clf2.fit(dataset.X, dataset.y);

    auto centroids1 = clf1.get_centroids();
    auto centroids2 = clf2.get_centroids();

    ASSERT_EQ(centroids1.size(), centroids2.size());

    for (size_t i = 0; i < centroids1.size(); ++i) {
        ASSERT_EQ(centroids1[i].size(), centroids2[i].size());
        for (size_t j = 0; j < centroids1[i].size(); ++j) {
            EXPECT_FLOAT_EQ(centroids1[i][j], centroids2[i][j]);
        }
    }
}

// CUDA CORRECTNESS TESTS

TEST(CUDA_Correctness, SameResultsAsCPU_SmallDataset) {
    auto dataset = generate_easy_realistic_dataset(10, 2, 4, 100);
    
    MinimumDistanceClassifier clf_cpu(false);
    MinimumDistanceClassifier clf_gpu(true);

    if (!clf_gpu.is_using_cuda()) {
        GTEST_SKIP() << "CUDA not available";
    }

    clf_cpu.fit(dataset.X, dataset.y);
    clf_gpu.fit(dataset.X, dataset.y);

    auto prediction_cpu = clf_cpu.predict(dataset.X);
    auto prediction_gpu = clf_gpu.predict(dataset.X);

    EXPECT_EQ(prediction_cpu, prediction_gpu);

    float accuracy_cpu = calculate_accuracy(dataset.y, prediction_cpu);
    float accuracy_gpu = calculate_accuracy(dataset.y, prediction_gpu);

    EXPECT_FLOAT_EQ(accuracy_cpu, accuracy_gpu);
}

TEST(CUDA_Correctness, SameResultsAsCPU_MediumDataset) {
    auto dataset = generate_medium_realistic_dataset(100, 5, 12, 200);
    
    MinimumDistanceClassifier clf_cpu(false);
    MinimumDistanceClassifier clf_gpu(true);

    if (!clf_gpu.is_using_cuda()) {
        GTEST_SKIP() << "CUDA not available";
    }

    clf_cpu.fit(dataset.X, dataset.y);
    clf_gpu.fit(dataset.X, dataset.y);

    auto prediction_cpu = clf_cpu.predict(dataset.X);
    auto prediction_gpu = clf_gpu.predict(dataset.X);

    EXPECT_EQ(prediction_cpu, prediction_gpu);

    float accuracy_cpu = calculate_accuracy(dataset.y, prediction_cpu);
    float accuracy_gpu = calculate_accuracy(dataset.y, prediction_gpu);

    EXPECT_FLOAT_EQ(accuracy_cpu, accuracy_gpu);
}

TEST(CUDA_Correctness, SameResultsAsCPU_LargeDataset) {
    auto dataset = generate_easy_realistic_dataset(10000, 20, 40, 300);
    
    MinimumDistanceClassifier clf_cpu(false);
    MinimumDistanceClassifier clf_gpu(true);

    if (!clf_gpu.is_using_cuda()) {
        GTEST_SKIP() << "CUDA not available";
    }

    clf_cpu.fit(dataset.X, dataset.y);
    clf_gpu.fit(dataset.X, dataset.y);

    auto prediction_cpu = clf_cpu.predict(dataset.X);
    auto prediction_gpu = clf_gpu.predict(dataset.X);

    int matches = 0;
    for (size_t i = 0; i < prediction_cpu.size(); ++i) {
        if (prediction_cpu[i] == prediction_gpu[i]) {
            matches++;
        }
    }
    float match_rate = static_cast<float>(matches) / prediction_cpu.size();
    
    EXPECT_GT(match_rate, 0.98f); 
}

// CUDA PRECISION TESTS

TEST(CUDA_Precision, NoSignificantFloatingPointErrors) {
    auto dataset = generate_medium_realistic_dataset(100, 5, 50, 400);
    
    MinimumDistanceClassifier clf_cpu(false);
    MinimumDistanceClassifier clf_gpu(true);

    if (!clf_gpu.is_using_cuda()) {
        GTEST_SKIP() << "CUDA not available";
    }

    clf_cpu.fit(dataset.X, dataset.y);
    clf_gpu.fit(dataset.X, dataset.y);

    auto prediction_cpu = clf_cpu.predict(dataset.X);
    auto prediction_gpu = clf_gpu.predict(dataset.X);

    int differences = 0;
    for (size_t i = 0; i < prediction_cpu.size(); ++i) {
        if (prediction_cpu[i] != prediction_gpu[i]) {
            differences++;
        }
    }

    float error_rate = static_cast<float>(differences) / prediction_cpu.size();

    EXPECT_LT(error_rate, 0.02f);
}

TEST(CUDA_Precision, CentroidsWithinTolerance) {
    auto dataset = generate_medium_realistic_dataset(200, 10, 30, 500);
    
    MinimumDistanceClassifier clf_cpu(false);
    MinimumDistanceClassifier clf_gpu(true);

    if (!clf_gpu.is_using_cuda()) {
        GTEST_SKIP() << "CUDA not available";
    }

    clf_cpu.fit(dataset.X, dataset.y);
    clf_gpu.fit(dataset.X, dataset.y);

    auto centroids_cpu = clf_cpu.get_centroids();
    auto centroids_gpu = clf_gpu.get_centroids();

    float max_diff = 0.0f;
    ASSERT_EQ(centroids_cpu.size(), centroids_gpu.size());
    for (size_t i = 0; i < centroids_cpu.size(); ++i) {
        ASSERT_EQ(centroids_cpu[i].size(), centroids_gpu[i].size());
        for (size_t f = 0; f < centroids_cpu[i].size(); ++f) {
            float diff = std::abs(centroids_cpu[i][f] - centroids_gpu[i][f]);
            max_diff = std::max(diff, max_diff);
            EXPECT_NEAR(centroids_cpu[i][f], centroids_gpu[i][f], 1e-4);
        }
    }
    EXPECT_LT(max_diff, 1e-3);
}

TEST(CUDA_Precision, PredictionsExactMatch) {
    auto dataset = generate_medium_realistic_dataset(150, 8, 20, 600);
    
    MinimumDistanceClassifier clf_cpu(false);
    MinimumDistanceClassifier clf_gpu(true);

    if (!clf_gpu.is_using_cuda()) {
        GTEST_SKIP() << "CUDA not available";
    }

    clf_cpu.fit(dataset.X, dataset.y);
    clf_gpu.fit(dataset.X, dataset.y);

    auto prediction_cpu = clf_cpu.predict(dataset.X);
    auto prediction_gpu = clf_gpu.predict(dataset.X);

    ASSERT_EQ(prediction_cpu.size(), prediction_gpu.size());

    int matches = 0;
    for (size_t i = 0; i < prediction_cpu.size(); ++i) {
        if (prediction_cpu[i] == prediction_gpu[i]) {
            matches++;
        }
    }

    float match_rate = static_cast<float>(matches) / prediction_cpu.size();

    EXPECT_GT(match_rate, 0.98f);  // At least 98% exact matches
}

// CUDA DETERMINISM TESTS

TEST(CUDA_Determinism, RepeatedRunsSameResults) {
    auto dataset = generate_medium_realistic_dataset(100, 5, 25, 42);
    
    MinimumDistanceClassifier clf_cuda(true);

    if (!clf_cuda.is_using_cuda()) {
        GTEST_SKIP() << "CUDA not available";
    }

    clf_cuda.fit(dataset.X, dataset.y);

    std::vector<std::vector<int>> all_predictions;
    for (int run = 0; run < 10; ++run) {
        all_predictions.push_back(clf_cuda.predict(dataset.X));
    }

    for (size_t run = 1; run < all_predictions.size(); ++run) {
        EXPECT_EQ(all_predictions[0], all_predictions[run])
            << "Run " << run << " produced different results";
    }
}

// CUDA EDGE CASES TESTS

TEST(CUDA_EdgeCases, VerySmallDataset_1Sample) {
    std::vector<std::vector<float>> X = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}};
    std::vector<int> y = {0};

    MinimumDistanceClassifier clf_cpu(false);
    MinimumDistanceClassifier clf_gpu(true);

    if (!clf_gpu.is_using_cuda()) {
        GTEST_SKIP() << "CUDA not available";
    }

    EXPECT_NO_THROW(clf_cpu.fit(X, y));
    EXPECT_NO_THROW(clf_gpu.fit(X, y));

    std::vector<std::vector<float>> X_test = {
        {1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f},
        {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}};

    auto pred_cpu = clf_cpu.predict(X_test);
    auto pred_gpu = clf_gpu.predict(X_test);

    EXPECT_EQ(pred_cpu, pred_gpu);

    for (const auto &p : pred_gpu) {
        EXPECT_EQ(p, 0);
    }
}

TEST(CUDA_EdgeCases, VerySmallDataset_10Samples) {
    auto dataset = generate_easy_realistic_dataset(5, 2, 6, 777);
    
    MinimumDistanceClassifier clf_cpu(false);
    MinimumDistanceClassifier clf_gpu(true);

    if (!clf_gpu.is_using_cuda()) {
        GTEST_SKIP() << "CUDA not available";
    }

    clf_cpu.fit(dataset.X, dataset.y);
    clf_gpu.fit(dataset.X, dataset.y);

    auto pred_cpu = clf_cpu.predict(dataset.X);
    auto pred_gpu = clf_gpu.predict(dataset.X);

    EXPECT_EQ(pred_cpu, pred_gpu);
}

TEST(CUDA_EdgeCases, BlockSizedDataset_256Samples) {
    auto dataset = generate_easy_realistic_dataset(128, 2, 8, 888);
    
    MinimumDistanceClassifier clf_cpu(false);
    MinimumDistanceClassifier clf_gpu(true);

    if (!clf_gpu.is_using_cuda()) {
        GTEST_SKIP() << "CUDA not available";
    }

    clf_cpu.fit(dataset.X, dataset.y);
    clf_gpu.fit(dataset.X, dataset.y);

    auto pred_cpu = clf_cpu.predict(dataset.X);
    auto pred_gpu = clf_gpu.predict(dataset.X);

    EXPECT_EQ(pred_cpu, pred_gpu);
    
    float gpu_accuracy = calculate_accuracy(dataset.y, pred_gpu);
    EXPECT_GT(gpu_accuracy, 0.75f);
}

TEST(CUDA_EdgeCases, JustOverBlockSize_258Samples) {
    auto dataset = generate_easy_realistic_dataset(129, 2, 8, 999);
    
    MinimumDistanceClassifier clf_cpu(false);
    MinimumDistanceClassifier clf_gpu(true);

    if (!clf_gpu.is_using_cuda()) {
        GTEST_SKIP() << "CUDA not available";
    }

    clf_cpu.fit(dataset.X, dataset.y);
    clf_gpu.fit(dataset.X, dataset.y);

    auto pred_cpu = clf_cpu.predict(dataset.X);
    auto pred_gpu = clf_gpu.predict(dataset.X);

    EXPECT_EQ(pred_cpu, pred_gpu);
    
    float gpu_accuracy = calculate_accuracy(dataset.y, pred_gpu);
    EXPECT_GT(gpu_accuracy, 0.75f);
}

TEST(CUDA_EdgeCases, ExactlyGridSize_65536Samples) {
    auto dataset = generate_easy_realistic_dataset(16384, 4, 16, 1000);
    
    MinimumDistanceClassifier clf_cpu(false);
    MinimumDistanceClassifier clf_gpu(true);

    if (!clf_gpu.is_using_cuda()) {
        GTEST_SKIP() << "CUDA not available";
    }

    EXPECT_NO_THROW(clf_cpu.fit(dataset.X, dataset.y));
    EXPECT_NO_THROW(clf_gpu.fit(dataset.X, dataset.y));

    // Test on subset
    Dataset test_subset;
    test_subset.X.assign(dataset.X.begin(), dataset.X.begin() + 1000);
    test_subset.y.assign(dataset.y.begin(), dataset.y.begin() + 1000);

    auto pred_cpu = clf_cpu.predict(test_subset.X);
    auto pred_gpu = clf_gpu.predict(test_subset.X);

    int matches = 0;
    for (size_t i = 0; i < pred_cpu.size(); ++i) {
        if (pred_cpu[i] == pred_gpu[i]) matches++;
    }
    float match_rate = static_cast<float>(matches) / pred_cpu.size();
    
    EXPECT_GT(match_rate, 0.95f);
    
    float cpu_accuracy = calculate_accuracy(test_subset.y, pred_cpu);
    float gpu_accuracy = calculate_accuracy(test_subset.y, pred_gpu);

    EXPECT_GT(cpu_accuracy, 0.75f);
    EXPECT_GT(gpu_accuracy, 0.75f);
}

TEST(CUDA_EdgeCases, VeryLargeDataset_1MSamples) {
    auto dataset = generate_medium_realistic_dataset(50000, 20, 50, 500);
    
    MinimumDistanceClassifier clf_gpu(true);

    if (!clf_gpu.is_using_cuda()) {
        GTEST_SKIP() << "CUDA not available";
    }

    EXPECT_NO_THROW(clf_gpu.fit(dataset.X, dataset.y));

    Dataset test_subset;
    test_subset.X.assign(dataset.X.begin(), dataset.X.begin() + 2000);
    test_subset.y.assign(dataset.y.begin(), dataset.y.begin() + 2000);

    auto predictions_gpu = clf_gpu.predict(test_subset.X);

    std::set<int> unique_predictions(predictions_gpu.begin(), predictions_gpu.end());
    EXPECT_GE(unique_predictions.size(), 12);

    std::vector<int> class_counts(20, 0);
    for (const auto &p : predictions_gpu) {
        EXPECT_GE(p, 0);
        EXPECT_LT(p, 20);
        class_counts[p]++;
    }

    int total = std::accumulate(class_counts.begin(), class_counts.end(), 0);
    EXPECT_EQ(total, 2000);

    int classes_with_predictions = 0;
    for (int c = 0; c < 20; ++c) {
        if (class_counts[c] > 0) {
            classes_with_predictions++;
        }
    }
    EXPECT_GE(classes_with_predictions, 12);

    float gpu_accuracy = calculate_accuracy(test_subset.y, predictions_gpu);
    EXPECT_GT(gpu_accuracy, 0.45f);  // Medium difficulty dataset

    std::cout << "1M samples (medium realistic) - GPU Accuracy: " << (gpu_accuracy * 100) << "%"
              << std::endl;
}

// MAIN

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
