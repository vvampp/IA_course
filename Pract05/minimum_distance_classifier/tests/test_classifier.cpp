#include "classifier.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <random>
#include <stdexcept>
#include <vector>

using namespace mdc;

struct Dataset{
    std::vector<std::vector<float>> X;
    std::vector<int> y;
};

Dataset generate_clusters(int n_samples_per_class, int n_classes,
                          int n_features, float separation = 5.0f,
                          unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    Dataset dataset;
    dataset.X.reserve(n_samples_per_class * n_classes);
    dataset.y.reserve(n_samples_per_class * n_classes);
    
    for(int c = 0 ; c < n_classes; ++c){
        std::vector<float> center(n_features);
        for(int f = 0; f < n_features; ++f){
            center[f] = c * separation;
        }

        for(int i = 0; i < n_samples_per_class ; ++i){
            std::vector<float> sample(n_features);
            for(int f = 0 ; f < n_features ; ++f){
                sample[f] = center[f] + dist(rng);
            }
            dataset.X.push_back(sample);
            dataset.y.push_back(c);
        }
    }

    // shuffle dataset
    std::vector<size_t> indices(dataset.X.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    Dataset shuffled;
    shuffled.X.reserve(dataset.X.size());
    shuffled.y.reserve(dataset.y.size());

    for(const auto& idx : indices){
        shuffled.X.push_back(dataset.X[idx]);
        shuffled.y.push_back(dataset.y[idx]);
    }

    return shuffled;
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

// validation tests

TEST_F(MinimumDistanceClassifierTest, EmtpyData) {
    MinimumDistanceClassifier clf(false);
    std::vector<std::vector<float>> X_empty;
    std::vector<int> y_empty;

    EXPECT_THROW(clf.fit(X_empty, y_empty), std::invalid_argument);
}

TEST_F(MinimumDistanceClassifierTest, EmtpyTags) {
    MinimumDistanceClassifier clf(false);
    std::vector<int> y_empty;

    EXPECT_THROW(clf.fit(X_simple, y_empty), std::invalid_argument);
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
    EXPECT_THROW(clf.predict(X_simple), std::runtime_error);
}

TEST_F(MinimumDistanceClassifierTest, GetCentroidsBeforeFit) {
    MinimumDistanceClassifier clf(false);
    EXPECT_THROW(clf.get_centroids(), std::runtime_error);
}

TEST_F(MinimumDistanceClassifierTest, WrongFeatureCountOnPredict) {
    MinimumDistanceClassifier clf(false);
    clf.fit(X_simple, y_simple); // 2 features

    std::vector<std::vector<float>> X_wrong = {{1.0f, 2.0f, 3.0f}};
    EXPECT_THROW(clf.predict(X_wrong), std::invalid_argument);
}

TEST_F(MinimumDistanceClassifierTest, MaxClassesLimit) {
    MinimumDistanceClassifier clf(false);
    std::vector<std::vector<float>> X = {{1.0f, 2.0f}};
    std::vector<int> y = {1000000}; // high class id

    EXPECT_THROW(clf.fit(X, y), std::invalid_argument);
}

// accuracy tests
TEST(ClassifierAccuracyTests, WellSeparatedClusters2D) {
    auto dataset = generate_clusters(50,3,2,10.0f);

    MinimumDistanceClassifier clf(false);
    clf.fit(dataset.X, dataset.y);

    auto predictions = clf.predict(dataset.X);
    float accuracy = calculate_accuracy(dataset.y, predictions);

    EXPECT_GT(accuracy, 0.95f);
}

TEST(ClassifierAccuracyTests, WellSeparatedClusters10D) {
    auto dataset = generate_clusters(30, 4, 10, 15.0f);

    MinimumDistanceClassifier clf(false);
    clf.fit(dataset.X, dataset.y);

    auto predictions = clf.predict(dataset.X);
    float accuracy = calculate_accuracy(dataset.y, predictions);

    EXPECT_GT(accuracy, 0.90f);
}

TEST(ClassifierAccuracyTests, OverlappingClusters) {
    auto dataset = generate_clusters(50, 2, 2, 1.0f);

    MinimumDistanceClassifier clf(false);
    clf.fit(dataset.X, dataset.y);

    auto predictions = clf.predict(dataset.X);
    float accuracy = calculate_accuracy(dataset.y, predictions);

    EXPECT_GT(accuracy, 0.5f); // better than a coin toss
}

TEST(ClassifierAccuracyTests, HeavyWorkload) {
    auto dataset = generate_clusters(20000, 50, 120, 3.0f);

    MinimumDistanceClassifier clf(false);
    clf.fit(dataset.X, dataset.y);

    auto predictions = clf.predict(dataset.X);
    float accuracy = calculate_accuracy(dataset.y, predictions);

    EXPECT_GT(accuracy, 0.75f); // better than a coin toss
}

// edge case testing

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
    int n_classes = 100;
    auto dataset = generate_clusters(5, n_classes, 10, 20.0f);

    MinimumDistanceClassifier clf(false);
    EXPECT_NO_THROW(clf.fit(dataset.X, dataset.y));

    EXPECT_EQ(clf.get_n_classes(), n_classes);
}

TEST(ClassifierEdgeCases, HighDimensional) {
    int n_features = 500;
    auto dataset = generate_clusters(10, 3, n_features, 50.0f);

    MinimumDistanceClassifier clf(false);
    EXPECT_NO_THROW(clf.fit(dataset.X, dataset.y));

    EXPECT_EQ(clf.get_n_features(), n_features);
}

TEST(ClassifierEdgeCases, LargeDataset) {
    int n_samples_per_class = 1000;

    auto dataset = generate_clusters(n_samples_per_class, 5, 20, 10.0f);

    MinimumDistanceClassifier clf(false);
    EXPECT_NO_THROW(clf.fit(dataset.X, dataset.y));

    auto predictions = clf.predict(dataset.X);
    EXPECT_EQ(predictions.size(), n_samples_per_class * 5);
}

TEST(ClassifierEdgeCases, EmptyPredictionSet) {
    MinimumDistanceClassifier clf(false);
    std::vector<std::vector<float>> X_train = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    std::vector<int> y_train = {0, 1};

    EXPECT_NO_THROW(clf.fit(X_train, y_train));

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

// determinism tests
TEST(ClassifierDeterminism, ConsistentResults) {
    auto dataset = generate_clusters(50, 3, 5, 10.0f, 12345);

    MinimumDistanceClassifier clf1(false);
    MinimumDistanceClassifier clf2(false);

    clf1.fit(dataset.X, dataset.y);
    clf2.fit(dataset.X, dataset.y);

    auto pred1 = clf1.predict(dataset.X);
    auto pred2 = clf2.predict(dataset.X);

    EXPECT_EQ(pred1, pred2);
}

TEST(ClassifierDeterminism, ConsistentCentroids) {
    auto dataset = generate_clusters(50, 3, 5, 10.0f, 54321);

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

// CUDA tests

TEST(CUDA_Correctness, SameResultsAsCPU_SmallDataset) {
    auto dataset = generate_clusters(10, 2, 4);

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

    EXPECT_EQ(accuracy_cpu, accuracy_gpu);
}

TEST(CUDA_Correctness, SameResultsAsCPU_MediumDataset) {
    auto dataset = generate_clusters(100, 5, 12);

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

    EXPECT_EQ(accuracy_cpu, accuracy_gpu);
}

TEST(CUDA_Correctness, SameResultsAsCPU_LargeDataset) {
    auto dataset = generate_clusters(10000, 20, 40);

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

    EXPECT_EQ(accuracy_cpu, accuracy_gpu);
}

// CUDA presition

TEST(CUDA_Precision, NoSignificantFloatingPointErrors) {
    auto dataset = generate_clusters(100, 5, 50, 10.0f);

    MinimumDistanceClassifier clf_cpu(false);
    MinimumDistanceClassifier clf_gpu(true);

    if (!clf_gpu.is_using_cuda()) {
        GTEST_SKIP() << "CUDA not available";
    }

    clf_cpu.fit(dataset.X, dataset.y);
    clf_gpu.fit(dataset.X, dataset.y);

    auto prediction_cpu = clf_cpu.predict(dataset.X);
    auto prediction_gpu = clf_gpu.predict(dataset.X);

    // Calculate max difference in predictions
    int differences = 0;
    for (size_t i = 0; i < prediction_cpu.size(); ++i) {
        if (prediction_cpu[i] != prediction_gpu[i]) {
            differences++;
        }
    }

    float error_rate = static_cast<float>(differences) / prediction_cpu.size();

    EXPECT_LT(error_rate, 0.01f);
}

TEST(CUDA_Precision, CentroidsWithinTolerance) {
    auto dataset = generate_clusters(200, 10, 30, 5.0f);

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
    auto dataset = generate_clusters(150, 8, 20, 8.0f);

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

    EXPECT_GT(match_rate, 0.99f);
}

TEST(CUDA_Determinism, RepeatedRunsSameResults){
    auto dataset = generate_clusters(1000, 5, 24, 4.0f);
    auto dataset_test = generate_clusters(1000, 5, 24, 4.0f, 999);

    MinimumDistanceClassifier clf1(true);
    MinimumDistanceClassifier clf2(true);
    MinimumDistanceClassifier clf3(true);

    if(!clf1.is_using_cuda()){
        GTEST_SKIP() << "CUDA not available";
    }
    if(!clf2.is_using_cuda()){
        GTEST_SKIP() << "CUDA not available";
    }
    if(!clf3.is_using_cuda()){
        GTEST_SKIP() << "CUDA not available";
    }

    EXPECT_NO_THROW(clf1.fit(dataset.X,dataset.y));
    EXPECT_NO_THROW(clf2.fit(dataset.X,dataset.y));
    EXPECT_NO_THROW(clf3.fit(dataset.X,dataset.y));

    auto pred1 = clf1.predict(dataset_test.X);
    auto pred2 = clf2.predict(dataset_test.X);
    auto pred3 = clf3.predict(dataset_test.X);

    ASSERT_EQ(pred1.size(), pred2.size());
    ASSERT_EQ(pred2.size(), pred3.size());

    for(size_t i = 0; i < pred1.size(); ++i){
        EXPECT_EQ(pred1[i], pred2[i]);
        EXPECT_EQ(pred2[i], pred3[i]);
    }
}

// CUDA edge cases

TEST(CUDA_EdgeCases, VerySmallDataset_1Sample){
    std::vector<std::vector<float>> X = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}};
    std::vector<int> y = {0};

    MinimumDistanceClassifier clf_cpu(false);
    MinimumDistanceClassifier clf_gpu(true);

    if(!clf_gpu.is_using_cuda()){
        GTEST_SKIP() << "CUDA not available";
    }

    clf_cpu.fit(X, y);
    clf_gpu.fit(X, y);

    std::vector<std::vector<float>> X_test = {
        { 1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f},  // close to class
        { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},  // exact
        {10.0f, 20.0, 30.0f, 40.0f, 50.0f, 60.0f}   // away from class
    };

    auto predict_cpu = clf_cpu.predict(X_test);
    auto predict_gpu = clf_gpu.predict(X_test);

    EXPECT_EQ(predict_cpu, predict_cpu);

    for(const auto&p : predict_gpu) {
        EXPECT_EQ(p, 0);
    }

    std::map<int, int> class_distribution;
    for(const auto& p : predict_gpu){
        class_distribution[p]++;
    }

    EXPECT_EQ(class_distribution.size(), 1);
    EXPECT_EQ(class_distribution[0], 3);
}

TEST(CUDA_EdgeCases, VerySmallDataset_10Samples){
    auto dataset = generate_clusters(5, 2, 6, 5.0f);

    MinimumDistanceClassifier clf_cpu(false);
    MinimumDistanceClassifier clf_gpu(true);

    if(!clf_gpu.is_using_cuda()){
        GTEST_SKIP() << "CUDA not available";
    }

    clf_cpu.fit(dataset.X, dataset.y);
    clf_gpu.fit(dataset.X, dataset.y);

    auto dataset_test = generate_clusters(10, 2, 6, 3.0f, 9999);

    auto predictions_cpu = clf_cpu.predict(dataset_test.X);
    auto predictions_gpu = clf_gpu.predict(dataset_test.X);

    EXPECT_EQ(predictions_cpu, predictions_gpu);

    int count_class_0 = std::count(predictions_gpu.begin(), predictions_gpu.end(), 0);
    int count_class_1 = std::count(predictions_gpu.begin(), predictions_gpu.end(), 1);

    EXPECT_GT(count_class_0, 0); // predict some as 0 
    EXPECT_GT(count_class_1, 0); // predict some as 1
    EXPECT_EQ(count_class_0 + count_class_1, 20);

    float cpu_accuracy = calculate_accuracy(dataset_test.y, predictions_cpu);
    float gpu_accuracy = calculate_accuracy(dataset_test.y, predictions_gpu);
    
    EXPECT_GT(gpu_accuracy, 0.5f);
    EXPECT_EQ(cpu_accuracy, gpu_accuracy);
}

TEST(CUDA_EdgeCases, BlockSizedDataset_256Samples){
    auto dataset = generate_clusters(128, 2, 8, 8.0f, 200);

    MinimumDistanceClassifier clf_cpu(false);
    MinimumDistanceClassifier clf_gpu(true);

    if(!clf_gpu.is_using_cuda()){
        GTEST_SKIP() << "CUDA not available";
    }

    clf_cpu.fit(dataset.X, dataset.y);
    clf_gpu.fit(dataset.X, dataset.y);

    auto dataset_test = generate_clusters(200, 2, 8, 8.0f, 222);

    auto predictions_cpu = clf_cpu.predict(dataset_test.X);
    auto predictions_gpu = clf_gpu.predict(dataset_test.X);

    EXPECT_EQ(predictions_cpu, predictions_gpu);

    int count_class_0 = std::count(predictions_gpu.begin(), predictions_gpu.end(), 0);
    int count_class_1 = std::count(predictions_gpu.begin(), predictions_gpu.end(), 1);

    EXPECT_GT(count_class_0, 50); // predict significant as 0 
    EXPECT_GT(count_class_1, 50); // predict significant as 1
    EXPECT_EQ(count_class_0 + count_class_1, 400);

    float ratio = static_cast<float>(count_class_0) / (count_class_0 + count_class_1);

    EXPECT_GT(ratio, 0.3f);
    EXPECT_LT(ratio, 0.7f);

    float cpu_accuracy = calculate_accuracy(dataset_test.y, predictions_cpu);
    float gpu_accuracy = calculate_accuracy(dataset_test.y, predictions_gpu);
    
    EXPECT_GT(gpu_accuracy, 0.85f);
    EXPECT_EQ(cpu_accuracy, gpu_accuracy);
}


TEST(CUDA_EdgeCases, JustOverBlockSize_258Samples){
    auto dataset = generate_clusters(129, 2, 8, 8.0f, 300);

    MinimumDistanceClassifier clf_cpu(false);
    MinimumDistanceClassifier clf_gpu(true);

    if(!clf_gpu.is_using_cuda()){
        GTEST_SKIP() << "CUDA not available";
    }

    clf_cpu.fit(dataset.X, dataset.y);
    clf_gpu.fit(dataset.X, dataset.y);

    auto dataset_test = generate_clusters(200, 2, 8, 8.0f, 333);

    auto predictions_cpu = clf_cpu.predict(dataset_test.X);
    auto predictions_gpu = clf_gpu.predict(dataset_test.X);

    EXPECT_EQ(predictions_cpu, predictions_gpu);

    int count_class_0 = std::count(predictions_gpu.begin(), predictions_gpu.end(), 0);
    int count_class_1 = std::count(predictions_gpu.begin(), predictions_gpu.end(), 1);

    EXPECT_GT(count_class_0, 50); // predict significant as 0 
    EXPECT_GT(count_class_1, 50); // predict significant as 1
    EXPECT_EQ(count_class_0 + count_class_1, 400);

    float ratio = static_cast<float>(count_class_0) / (count_class_0 + count_class_1);

    EXPECT_GT(ratio, 0.3f);
    EXPECT_LT(ratio, 0.7f);

    float cpu_accuracy = calculate_accuracy(dataset_test.y, predictions_cpu);
    float gpu_accuracy = calculate_accuracy(dataset_test.y, predictions_gpu);
    
    EXPECT_GT(gpu_accuracy, 0.85f);
    EXPECT_EQ(cpu_accuracy, gpu_accuracy);
}

TEST(CUDA_EdgeCases, ExactlyGridSize_65536Samples){
    auto dataset = generate_clusters(16384, 4, 16, 15.0f, 400);

    MinimumDistanceClassifier clf_cpu(false);
    MinimumDistanceClassifier clf_gpu(true);

    if(!clf_gpu.is_using_cuda()){
        GTEST_SKIP() << "CUDA not available";
    }

    EXPECT_NO_THROW(clf_cpu.fit(dataset.X, dataset.y));
    EXPECT_NO_THROW(clf_gpu.fit(dataset.X, dataset.y));

    auto dataset_test = generate_clusters(500, 4, 16, 15.0f, 444);

    auto predictions_cpu = clf_cpu.predict(dataset_test.X);
    auto predictions_gpu = clf_gpu.predict(dataset_test.X);
    
    EXPECT_EQ(predictions_cpu, predictions_gpu);

    std::vector<int> class_counts(4,0);
    for(auto const&p : predictions_gpu){
        class_counts[p]++;
    }

    int total = std::accumulate(class_counts.begin(), class_counts.end(), 0);
    EXPECT_EQ(total, 2000);

    float cpu_accuracy = calculate_accuracy(dataset_test.y, predictions_cpu);
    float gpu_accuracy = calculate_accuracy(dataset_test.y, predictions_gpu);

    EXPECT_GT(cpu_accuracy, 0.75f);
    EXPECT_GT(gpu_accuracy, 0.75f);
    EXPECT_NEAR(cpu_accuracy, gpu_accuracy, 0.01f);
}

TEST(CUDA_EdgeCases, VeryLargeDataset_1MSamples){
    auto dataset = generate_clusters(50000, 20, 50, 20.0f, 500);

    MinimumDistanceClassifier clf_cpu(false);
    MinimumDistanceClassifier clf_gpu(true);

    if(!clf_gpu.is_using_cuda()){
        GTEST_SKIP() << "CUDA not available";
    }

    EXPECT_NO_THROW(clf_gpu.fit(dataset.X, dataset.y));

    auto dataset_test = generate_clusters(1000, 20, 50, 20.0f, 555);

    auto predictions_gpu = clf_gpu.predict(dataset_test.X);

    std::set<int> unique_predictions(predictions_gpu.begin(), predictions_gpu.end());
    EXPECT_GE(unique_predictions.size(), 16); // should preict most classes
    
    std::vector<int> class_counts(20,0);
    for(const auto& p : predictions_gpu){
        EXPECT_GE(p, 0);    // non negative prediction
        EXPECT_LT(p, 20);   // prediction within class range
        class_counts[p]++;  
    }
    
    int total = std::accumulate(class_counts.begin(), class_counts.end(), 0);
    EXPECT_EQ(total, 20000);

    int classes_with_predictions = 0;
    for(int c = 0; c < 20 ; ++c){
        if (class_counts[c] > 0) {
            classes_with_predictions++;
        }
    }

    EXPECT_GE(classes_with_predictions,16); // should predict most classes
    
    float gpu_accuracy = calculate_accuracy(dataset_test.y, predictions_gpu);
    EXPECT_GT(gpu_accuracy, 0.65f);

    std::cout << "1M samples - GPU Accuracy: " << (gpu_accuracy * 100) << "%" << std::endl;
}



// main

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
