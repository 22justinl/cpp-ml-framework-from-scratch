#include "doctest.h"

#include "core/tensor.h"
#include "ops/loss.h"
#include "utils/tensor_utils.h"

TEST_CASE("MSE Loss") {
    Tensor t1({1,2,3,4,5}, {1,5});
    Tensor t2({5,5,5,5,5}, {1,5});
    Tensor res1 = mse_loss(t1, t2);
    CHECK(check_tensor_equal(res1, Tensor({6},{1})));
}

TEST_CASE("Cross Entropy Loss") {
    Tensor t1({1,2,3,4,5}, {1,5});
    Tensor t2({5,5,5,5,5}, {1,5});
    Tensor t3({1,2,3,4,5,6,8,8,9,10,11,12,11,14,15}, {3,5});
    Tensor t4(1, {3,5});
    Tensor res1 = cross_entropy_loss(t1, t2);
    Tensor res2 = cross_entropy_loss(t3, t4);
    CHECK(check_tensor_equal(res1, Tensor({61.2979},{1}), 1e-4));
    CHECK(check_tensor_equal(res2, Tensor({12.5523},{1}), 1e-4));
}

TEST_CASE("Cross Entropy Loss (target indices)") {
    Tensor t1({1,2,3,4,5}, {1,5});
    Tensor t2({0}, {1});
    Tensor t3({1,2,3,4,5,6,8,8,9,10,11,12,11,14,15}, {3,5});
    Tensor t4(1, {3});
    Tensor res1 = cross_entropy_loss(t1, t2);
    CHECK(check_tensor_equal(res1, Tensor({4.4519},{1}), 1e-4));
}
