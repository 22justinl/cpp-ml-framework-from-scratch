#include "doctest.h"

#include "core/tensor.h"
#include "ops/reduction_ops.h"
#include "utils/tensor_utils.h"

TEST_CASE("Tensor sum") {
    Tensor t1({1,2,3,4,5,6}, {2,3});
    Tensor res1 = sum(t1);
    Tensor res2 = sum(t1, 0);
    Tensor res3 = sum(t1, 1);
    CHECK(check_tensor_equal(res1, Tensor({21}, {1})));
    CHECK(check_tensor_equal(res2, Tensor({5,7,9}, {3})));
    CHECK(check_tensor_equal(res3, Tensor({6,15}, {2})));
}
TEST_CASE("Tensor mean") {
    Tensor t1({1,2,3,4,5,6}, {2,3});
    Tensor res1 = mean(t1);
    Tensor res2 = mean(t1, 0);
    Tensor res3 = mean(t1, 1);
    CHECK(check_tensor_equal(res1, Tensor({21.0/6.0}, {1})));
    CHECK(check_tensor_equal(res2, Tensor({5.0/2,7.0/2,9.0/2}, {3})));
    CHECK(check_tensor_equal(res3, Tensor({6.0/3.0,15.0/3.0}, {2})));
}
TEST_CASE("Tensor max") {
    Tensor t1({1,2,3,4,5,6}, {2,3});
    Tensor t2({1,2,3,4,5,6}, {6});
    Tensor res1 = max(t1);
    Tensor res2 = max(t1, 0);
    Tensor res3 = max(t1, 1);
    Tensor res4 = max(t2);
    CHECK(check_tensor_equal(res1, Tensor({6}, {1})));
    CHECK(check_tensor_equal(res2, Tensor({4,5,6}, {3})));
    CHECK(check_tensor_equal(res3, Tensor({3, 6}, {2})));
    CHECK(check_tensor_equal(res4, Tensor({6}, {1})));
}
TEST_CASE("Tensor min") {
    Tensor t1({1,2,3,4,5,6}, {2,3});
    Tensor t2({1,2,3,4,5,6}, {6});
    Tensor res1 = min(t1);
    Tensor res2 = min(t1, 0);
    Tensor res3 = min(t1, 1);
    Tensor res4 = min(t2);
    CHECK(check_tensor_equal(res1, Tensor({1}, {1})));
    CHECK(check_tensor_equal(res2, Tensor({1,2,3}, {3})));
    CHECK(check_tensor_equal(res3, Tensor({1,4}, {2})));
    CHECK(check_tensor_equal(res4, Tensor({1}, {1})));
}
