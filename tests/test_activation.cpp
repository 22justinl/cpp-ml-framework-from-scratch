#include "doctest.h"

#include "core/tensor.h"
#include "ops/activation.h"
#include "utils/tensor_utils.h"

#include <cmath>
#include <numbers>

TEST_CASE("Sigmoid") {
    Tensor t1({-5,-2.5, 0, 2.5, 5}, {5});
    Tensor res1 = sigmoid(t1);
    Tensor expected1 = Tensor({float(1.0-1.0/(1.0+std::exp(-5))),float(1.0-1.0/(1.0+std::expf(-2.5))), 0.5, float(1.0/(1.0+std::exp(-2.5))), float(1.0/(1.0+std::exp(-5)))}, {5});
    CHECK(check_tensor_equal(res1, expected1));
}

TEST_CASE("ReLU") {
    Tensor t1({-50,-20,5,0,-000.1, 50, 20, -5, -0, 0.0001}, {2,5});
    Tensor res1 = relu(t1);
    Tensor expected1({0,0,5,0,0, 50, 20, 0, 0, 0.0001}, {2,5});
    CHECK(check_tensor_equal(res1, expected1));
}

TEST_CASE("Softmax") {
    Tensor t1({0, 0, 1, 0, 0}, {5});
    Tensor res1 = softmax(t1);
    float denom = 4.0+std::numbers::e_v<float>;
    Tensor expected1({1.f/denom, 1.f/denom, std::numbers::e_v<float>/denom, 1.f/denom, 1.f/denom}, {5});
    CHECK(check_tensor_equal(res1, expected1));
}
