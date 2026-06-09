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
    Tensor res1 = softmax(t1, 0);
    float denom = 4.0+std::numbers::e_v<float>;
    Tensor expected1({1.f/denom, 1.f/denom, std::numbers::e_v<float>/denom, 1.f/denom, 1.f/denom}, {5});
    CHECK(check_tensor_equal(res1, expected1));
}

TEST_CASE("Softmax (nD)") {
    Tensor t1({1,2,3,4,5,6}, {2,3});
    Tensor res1 = softmax(t1, 0);
    Tensor res2 = softmax(t1, 1);
    Tensor expected1({0.0474, 0.0474, 0.0474, 0.9526, 0.9526, 0.9526}, {2,3});
    Tensor expected2({0.0900, 0.2447, 0.6652, 0.0900, 0.2447, 0.6652}, {2,3});
    CHECK(check_tensor_equal(res1, expected1, 1e-3));
    CHECK(check_tensor_equal(res2, expected2, 1e-3));

    Tensor t2({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, {2,3,4});
    Tensor res3 = softmax(t2, 2);
    Tensor expected3({
            0.0321, 0.0871, 0.2369, 0.6439,
            0.0321, 0.0871, 0.2369, 0.6439,
            0.0321, 0.0871, 0.2369, 0.6439,

            0.0321, 0.0871, 0.2369, 0.6439,
            0.0321, 0.0871, 0.2369, 0.6439,
            0.0321, 0.0871, 0.2369, 0.6439}, {2,3,4});
    CHECK(check_tensor_equal(res3, expected3, 1e-3));
}
