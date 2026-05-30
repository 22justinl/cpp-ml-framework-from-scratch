#include "doctest.h"

#include "core/tensor.h"
#include "ops/reduction_ops.h"
#include "utils/tensor_utils.h"

TEST_CASE("Tensor sum autograd") {
    Tensor t1({1,2,3,4,5,6}, {2,3});
    Tensor res1 = sum(t1);
    Tensor res2 = sum(t1, 0);
    Tensor res3 = sum(t1, 1);
    CHECK(check_tensor_equal(res1, Tensor({1,1,1,1,1,1}, {2,3})));
    CHECK(check_tensor_equal(res2, Tensor({1,1,1,1,1,1}, {2,3})));
    CHECK(check_tensor_equal(res3, Tensor({1,1,1,1,1,1}, {2,3})));
}
TEST_CASE("Tensor mean") {
    Tensor t1({1,2,3,4,5,6}, {2,3});
    Tensor res1 = mean(t1);
    Tensor res2 = mean(t1, 0);
    Tensor res3 = mean(t1, 1);
    CHECK(check_tensor_equal(res1, Tensor({1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6}, {2,3})));
    CHECK(check_tensor_equal(res2, Tensor({1.0/2,1.0/2,1.0/2,1.0/2,1.0/2,1.0/2}, {2,3})));
    CHECK(check_tensor_equal(res3, Tensor({1.0/3,1.0/3,1.0/3,1.0/3,1.0/3,1.0/3}, {2,3})));
}
