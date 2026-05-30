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

}
TEST_CASE("Tensor max") {

}
TEST_CASE("Tensor min") {

}
