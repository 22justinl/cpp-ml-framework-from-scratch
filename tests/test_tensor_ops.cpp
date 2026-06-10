#include "doctest.h"

#include "core/tensor.h"
#include "ops/tensor_ops.h"
#include "utils/tensor_utils.h"

TEST_CASE("Tensor transpose") {
    Tensor t11({0,1,2,3,4},{5,1});
    Tensor t12({0,1,2,3,4},{1,5});
    Tensor t2({0,1,2,3,4,5,6,7,8,9,10,11}, {3,4});
    Tensor res11 = transpose(t11, 0, 1);
    Tensor res12 = transpose(t12, 0, 1);
    Tensor res2 = transpose(t2, 0, 1);
    CHECK(check_tensor_equal(res11, Tensor({0,1,2,3,4}, {1,5})));
    CHECK(check_tensor_equal(res12, Tensor({0,1,2,3,4}, {5,1})));
    CHECK(check_tensor_equal(res2, Tensor({0,4,8,1,5,9,2,6,10,3,7,11}, {4,3})));
}

TEST_CASE("Tensor transpose (nD)") {
    Tensor t1({
            1,2,3,4,
            5,6,7,8,
            9,10,11,12,

            13,14,15,16,
            17,18,19,20,
            21,22,23,24},{2,3,4});
    Tensor res1 = transpose(t1, 1, 2);
    Tensor res2 = transpose(t1, 0, 1);
    Tensor res3 = transpose(t1, 0, 2);
    CHECK(check_tensor_equal(res1, Tensor({
                    1,5,9,
                    2,6,10,
                    3,7,11,
                    4,8,12,

                    13,17,21,
                    14,18,22,
                    15,19,23,
                    16,20,24}, {2,4,3})));
    CHECK(check_tensor_equal(res2, Tensor({
                    1,2,3,4,
                    13,14,15,16,

                    5,6,7,8,
                    17,18,19,20,

                    9,10,11,12,
                    21,22,23,24}, {3,2,4})));
    CHECK(check_tensor_equal(res3, Tensor({
                    1,13,
                    5,17,
                    9,21,

                    2,14,
                    6,18,
                    10,22,

                    3,15,
                    7,19,
                    11,23,

                    4,16,
                    8,20,
                    12,24}, {4,3,2})));
}
