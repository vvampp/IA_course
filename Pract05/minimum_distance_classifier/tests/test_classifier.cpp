#include "classifier.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace mdc;

TEST(ConstructorTest, CreateWithoutCUDA) {
    MinimumDistanceClassifier clf(false);
    EXPECT_FALSE(clf.is_using_cuda());
    EXPECT_FALSE(clf.is_fitted());
}

TEST(ConstructorTest, CreateWithCUDA) {
    MinimumDistanceClassifier clf(true);

    EXPECT_FALSE(clf.is_fitted());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
