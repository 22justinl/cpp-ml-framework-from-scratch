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

TEST_CASE("Tensor sum (nD)") {
    Tensor t1({
            0,1,2,3,
            4,5,6,7,

            8,9,10,11,
            12,13,14,15,

            16,17,18,19,
            20,21,22,23}, {3,2,4});
    CHECK(check_tensor_equal(sum(t1, 0), Tensor({24,27,30,33,36,39,42,45},{2,4})));
    CHECK(check_tensor_equal(sum(t1, 1), Tensor({4,6,8,10,20,22,24,26,36,38,40,42},{3,4})));
    CHECK(check_tensor_equal(sum(t1, 2), Tensor({6,22,38,54,70,86},{3,2})));

    CHECK(check_tensor_equal(sum(t1, 0, true), Tensor({24,27,30,33,36,39,42,45},{1,2,4})));
    CHECK(check_tensor_equal(sum(t1, 1, true), Tensor({4,6,8,10,20,22,24,26,36,38,40,42},{3,1,4})));
    CHECK(check_tensor_equal(sum(t1, 2, true), Tensor({6,22,38,54,70,86},{3,2,1})));
}

TEST_CASE("Tensor max (nD)") {
    Tensor t1({
            0,1,2,3,
            4,5,6,7,

            8,9,10,11,
            12,13,14,15,

            16,17,18,19,
            20,21,22,23}, {3,2,4});
    CHECK(check_tensor_equal(max(t1, 0), Tensor({16,17,18,19,20,21,22,23},{2,4})));
    CHECK(check_tensor_equal(max(t1, 1), Tensor({4,5,6,7,12,13,14,15,20,21,22,23},{3,4})));
    CHECK(check_tensor_equal(max(t1, 2), Tensor({3,7,11,15,19,23},{3,2})));

    CHECK(check_tensor_equal(max(t1, 0, true), Tensor({16,17,18,19,20,21,22,23},{1,2,4})));
    CHECK(check_tensor_equal(max(t1, 1, true), Tensor({4,5,6,7,12,13,14,15,20,21,22,23},{3,1,4})));
    CHECK(check_tensor_equal(max(t1, 2, true), Tensor({3,7,11,15,19,23},{3,2,1})));
}
TEST_CASE("Tensor min (nD)") {
    Tensor t1({
            0,1,2,3,
            4,5,6,7,

            8,9,10,11,
            12,13,14,15,

            16,17,18,19,
            20,21,22,23}, {3,2,4});
    CHECK(check_tensor_equal(min(t1, 0), Tensor({0,1,2,3,4,5,6,7},{2,4})));
    CHECK(check_tensor_equal(min(t1, 1), Tensor({0,1,2,3,8,9,10,11,16,17,18,19},{3,4})));
    CHECK(check_tensor_equal(min(t1, 2), Tensor({0,4,8,12,16,20},{3,2})));

    CHECK(check_tensor_equal(min(t1, 0, true), Tensor({0,1,2,3,4,5,6,7},{1,2,4})));
    CHECK(check_tensor_equal(min(t1, 1, true), Tensor({0,1,2,3,8,9,10,11,16,17,18,19},{3,1,4})));
    CHECK(check_tensor_equal(min(t1, 2, true), Tensor({0,4,8,12,16,20},{3,2,1})));
}
