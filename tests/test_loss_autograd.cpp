#include "doctest.h"

#include "core/tensor.h"
#include "ops/loss.h"
#include "ops/reduction_ops.h"
#include "ops/math_ops.h"
#include "utils/tensor_utils.h"

TEST_CASE("MSE Loss autograd") {
    Tensor t1({1,2,3,4,5}, {5}, true);
    Tensor t2({5,5,5,5,5}, {5}, true);
    Tensor res1 = mse_loss(t1, t2);
    t1.zero_grad();
    t2.zero_grad();
    res1.backward();
    CHECK(check_tensor_equal(t1.grad(), Tensor({-1.6, -1.2, -0.8, -0.4, 0},{5})));
    CHECK(check_tensor_equal(t2.grad(), Tensor({1.6, 1.2, 0.8, 0.4, 0},{5})));
}

TEST_CASE("Cross Entropy Loss autograd") {
    Tensor t1({1,2,3,4,5}, {5}, true);
    Tensor t2({5,5,5,5,5}, {5}, true);
    Tensor res1 = cross_entropy_loss(t1, t2);
    t1.zero_grad();
    t2.zero_grad();
    res1.backward();
    print_tensor(t1.grad());
    print_tensor(t2.grad());
    CHECK(check_tensor_equal(t1.grad(), Tensor({-4.7086, -4.2079, -2.8468,  0.8530, 10.9102},{5}), 1e-4));
    CHECK(check_tensor_equal(t2.grad(), Tensor({4.4519, 3.4519, 2.4519, 1.4519, 0.4519},{5}), 1e-4));
}
