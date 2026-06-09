#include "doctest.h"

#include "core/tensor.h"
#include "ops/activation.h"
#include "utils/tensor_utils.h"

#include <cmath>

TEST_CASE("Sigmoid autograd") {
    Tensor t1({-5,-2.5, 0, 2.5, 5}, {5}, true);
    Tensor res1 = sigmoid(t1);
    t1.zero_grad();
    res1.backward();
    Tensor s = Tensor({float(1.0-1.0/(1.0+std::exp(-5))),float(1.0-1.0/(1.0+std::expf(-2.5))), 0.5, float(1.0/(1.0+std::exp(-2.5))), float(1.0/(1.0+std::exp(-5)))}, {5});
    Tensor expected1 = s*(Tensor(1, s.shape())-s);
    CHECK(check_tensor_equal(t1.grad(), expected1));
}

TEST_CASE("ReLU autograd") {
    Tensor t1({-50,-20,5,0,-000.1, 50, 20, -5, -0, 0.0001}, {2,5}, true);
    Tensor res1 = relu(t1);
    t1.zero_grad();
    res1.backward();
    Tensor expected1({0,0,1,0,0,1,1,0,0,1}, {2,5});
    CHECK(check_tensor_equal(t1.grad(), expected1));
}

TEST_CASE("Softmax autograd") {
    Tensor t1({1,2,3,4,5}, {5}, true);
    Tensor res1 = softmax(t1, 0);
    t1.zero_grad();
    res1.backward();
    Tensor expected1({0, 0, 0, 0, 0}, {5});
    CHECK(check_tensor_equal(t1.grad(), expected1));

    Tensor res2 = softmax(t1, 0);
    t1.zero_grad();
    res2.backward(Tensor({1,2,3,4,5},{5}));
    Tensor expected2({-0.0402, -0.0777, -0.1251, -0.1058,  0.3488}, {5});
    CHECK(check_tensor_equal(t1.grad(), expected2, 1e-4));
}

TEST_CASE("Softmax autograd (nD)") {
    Tensor t1({1,2,3,4,5,6}, {2,3}, true);
    Tensor res0 = softmax(t1, 0);
    t1.zero_grad();
    res0.backward();
    Tensor expected0({0, 0, 0, 0, 0, 0}, {2,3});
    CHECK(check_tensor_equal(t1.grad(), expected0));

    t1.zero_grad();
    res0.backward(Tensor({1,2,3,4,5,6}, {2,3}));
    Tensor expected1({-0.1355, -0.1355, -0.1355, 0.1355,  0.1355,  0.1355}, {2,3});
    CHECK(check_tensor_equal(t1.grad(), expected1, 1e-3));

    Tensor res2 = softmax(t1, 1);
    t1.zero_grad();
    res2.backward(Tensor({1,2,3,4,5,6},{2,3}));
    Tensor expected2({-0.1418, -0.1408,  0.2826, -0.1418, -0.1408,  0.2826}, {2,3});
    CHECK(check_tensor_equal(t1.grad(), expected2, 1e-4));

    Tensor t2({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, {2,3,4}, true);
    Tensor res3 = softmax(t2, 2);
    t2.zero_grad();
    res3.backward(Tensor({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, {2,3,4}));
    Tensor expected3({
            -0.0799, -0.1301, -0.1167,  0.3267,
            -0.0799, -0.1301, -0.1167,  0.3267,
            -0.0799, -0.1301, -0.1167,  0.3267,

            -0.0799, -0.1301, -0.1167,  0.3267,
            -0.0799, -0.1301, -0.1167,  0.3267,
            -0.0799, -0.1301, -0.1167,  0.3267}, {2,3,4});
    CHECK(check_tensor_equal(t2.grad(), expected3, 1e-4));
}
