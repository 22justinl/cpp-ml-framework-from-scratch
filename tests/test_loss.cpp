#include "doctest.h"

#include "core/tensor.h"
#include "ops/loss.h"
#include "utils/tensor_utils.h"

TEST_CASE("MSE Loss") {
    Tensor t1({1,2,3,4,5}, {5});
    Tensor t2({5,5,5,5,5}, {5});
    Tensor res1 = mse_loss(t1, t2);
    CHECK(check_tensor_equal(res1, Tensor({6},{1})));
}

TEST_CASE("Cross Entropy Loss") {
    Tensor t1({1,2,3,4,5}, {5});
    Tensor t2({5,5,5,5,5}, {5});
    Tensor res1 = cross_entropy_loss(t1, t2);
    CHECK(check_tensor_equal(res1, Tensor({61.2979},{1}), 1e-4));
}
