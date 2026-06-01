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
    Tensor res1 = softmax(t1);
    t1.zero_grad();
    res1.backward();
    Tensor expected1({0, 0, 0, 0, 0}, {5});
    CHECK(check_tensor_equal(t1.grad(), expected1));

    Tensor res2 = softmax(t1);
    t1.zero_grad();
    res2.backward(Tensor({1,2,3,4,5},{5}));
    Tensor expected2({-0.0402, -0.0777, -0.1251, -0.1058,  0.3488}, {5});
    CHECK(check_tensor_equal(t1.grad(), expected2, 1e-4));
}
