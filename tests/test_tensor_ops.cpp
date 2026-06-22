#include "doctest.h"

#include "core/tensor.h"
#include "ops/math_ops.h"
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

TEST_CASE("Tensor squeeze") {
    Tensor t1({1,2,3,4,5,6}, {1,2,1,1,3,1});
    Tensor res0 = squeeze(t1, 0);
    Tensor res1 = squeeze(res0, 0);
    Tensor res2 = squeeze(res1, 4);
    Tensor res3 = squeeze(res2, 2);
    Tensor res4 = squeeze(res3, 1);
    CHECK(check_tensor_equal(res0, Tensor({1,2,3,4,5,6},{2,1,1,3,1})));
    CHECK(check_tensor_equal(res1, Tensor({1,2,3,4,5,6},{2,1,1,3,1})));
    CHECK(check_tensor_equal(res2, Tensor({1,2,3,4,5,6},{2,1,1,3})));
    CHECK(check_tensor_equal(res3, Tensor({1,2,3,4,5,6},{2,1,3})));
    CHECK(check_tensor_equal(res4, Tensor({1,2,3,4,5,6},{2,3})));

    // non-continguous tensors
    Tensor t2 = transpose(Tensor({1, 2, 3, 4, 5, 6}, {2,3,1}), 0, 1);
    Tensor t3 = transpose(Tensor({1, 2, 3, 4, 5, 6}, {2,1,3}), 0, 2);
    Tensor t4 = transpose(Tensor({1, 2, 3, 4, 5, 6}, {1,2,3}), 1, 2);
    Tensor res5 = squeeze(t2, 2);
    Tensor res6 = squeeze(t3, 1);
    Tensor res7 = squeeze(t4, 0);
    CHECK(check_tensor_equal(res5, Tensor({1,4,2,5,3,6},{3,2})));
    CHECK(check_tensor_equal(res6, Tensor({1,4,2,5,3,6},{3,2})));
    CHECK(check_tensor_equal(res7, Tensor({1,4,2,5,3,6},{3,2})));
}

TEST_CASE("Tensor unsqueeze") {
    Tensor t1({1,2,3,4,5,6},{3,2});
    Tensor res0 = unsqueeze(t1, 0);
    Tensor res1 = unsqueeze(res0, 2);
    Tensor res2 = unsqueeze(res1, 2);
    Tensor res3 = unsqueeze(res2, 5);
    CHECK(check_tensor_equal(res0, Tensor({1,2,3,4,5,6},{1,3,2})));
    CHECK(check_tensor_equal(res1, Tensor({1,2,3,4,5,6},{1,3,1,2})));
    CHECK(check_tensor_equal(res2, Tensor({1,2,3,4,5,6},{1,3,1,1,2})));
    CHECK(check_tensor_equal(res3, Tensor({1,2,3,4,5,6},{1,3,1,1,2,1})));

    // non-contiguous tensors
    Tensor t2 = transpose(Tensor({1,2,3,4,5,6}, {2,3}), 0, 1);
    Tensor res4 = unsqueeze(t2, 0);
    Tensor res5 = unsqueeze(t2, 1);
    Tensor res6 = unsqueeze(t2, 2);
    CHECK(check_tensor_equal(res4, Tensor({1,4,2,5,3,6},{1,3,2})));
    CHECK(check_tensor_equal(res5, Tensor({1,4,2,5,3,6},{3,1,2})));
    CHECK(check_tensor_equal(res6, Tensor({1,4,2,5,3,6},{3,2,1})));
}

TEST_CASE("Tensor reshape (valid view: no copy)") {
    Tensor t1({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, {2,3,4});
    Tensor res1 = reshape(t1, {4,2,3});
    Tensor res2 = reshape(t1, {2,2,2,3});
    Tensor res3 = reshape(t1, {2,1,2,3,2});
    Tensor res4 = reshape(t1, {3,8});
    CHECK(check_tensor_equal(res1, Tensor({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, {4,2,3})));
    CHECK(check_tensor_equal(res2, Tensor({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, {2,2,2,3})));
    CHECK(check_tensor_equal(res3, Tensor({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, {2,1,2,3,2})));
    CHECK(check_tensor_equal(res4, Tensor({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, {3,8})));
    CHECK(t1.impl()->storage == res1.impl()->storage);
    CHECK(t1.impl()->storage == res2.impl()->storage);
    CHECK(t1.impl()->storage == res3.impl()->storage);
    CHECK(t1.impl()->storage == res4.impl()->storage);

    // non-contiguous tensors
    Tensor t2({
            1,2,3,
            4,5,6,
            7,8,9,
            10,11,12,

            13,14,15,
            16,17,18,
            19,20,21,
            22,23,24,

            25,26,27,
            28,29,30,
            31,32,33,
            34,35,36,


            37,38,39,
            40,41,42,
            43,44,45,
            46,47,48,

            49,50,51,
            52,53,54,
            55,56,57,
            58,59,60,

            61,62,63,
            64,65,66,
            67,68,69,
            70,71,72},{2,3,4,3});
    Tensor t3 = transpose(t2, 2, 3);
    std::vector<float> expected_data{
        1,  4,  7, 10,  2,  5,  8, 11,  3,  6,  9, 12, 13, 16, 19, 22, 14, 17,
        20, 23, 15, 18, 21, 24, 25, 28, 31, 34, 26, 29, 32, 35, 27, 30, 33, 36,
        37, 40, 43, 46, 38, 41, 44, 47, 39, 42, 45, 48, 49, 52, 55, 58, 50, 53,
        56, 59, 51, 54, 57, 60, 61, 64, 67, 70, 62, 65, 68, 71, 63, 66, 69, 72
    };
    Tensor res5 = reshape(t3, {6,3,4});
    Tensor res6 = reshape(t3, {2,3,3,1,4});
    Tensor res7 = reshape(t3, {6,3,2,2});
    CHECK(check_tensor_equal(res5, Tensor(expected_data, {6,3,4})));
    CHECK(check_tensor_equal(res6, Tensor(expected_data, {2,3,3,1,4})));
    CHECK(check_tensor_equal(res7, Tensor(expected_data, {6,3,2,2})));
    CHECK(t3.impl()->storage == res5.impl()->storage);
    CHECK(t3.impl()->storage == res6.impl()->storage);
    CHECK(t3.impl()->storage == res7.impl()->storage);
}

TEST_CASE("Tensor reshape (invalid view: copy)") {
    Tensor t1({
            1,2,3,
            4,5,6,
            7,8,9,
            10,11,12,

            13,14,15,
            16,17,18,
            19,20,21,
            22,23,24,

            25,26,27,
            28,29,30,
            31,32,33,
            34,35,36,


            37,38,39,
            40,41,42,
            43,44,45,
            46,47,48,

            49,50,51,
            52,53,54,
            55,56,57,
            58,59,60,

            61,62,63,
            64,65,66,
            67,68,69,
            70,71,72},{2,3,4,3});
    Tensor t2 = transpose(t1, 2, 3);
    std::vector<float> expected_data{
        1,  4,  7, 10,  2,  5,  8, 11,  3,  6,  9, 12, 13, 16, 19, 22, 14, 17,
        20, 23, 15, 18, 21, 24, 25, 28, 31, 34, 26, 29, 32, 35, 27, 30, 33, 36,
        37, 40, 43, 46, 38, 41, 44, 47, 39, 42, 45, 48, 49, 52, 55, 58, 50, 53,
        56, 59, 51, 54, 57, 60, 61, 64, 67, 70, 62, 65, 68, 71, 63, 66, 69, 72
    };
    Tensor res1 = reshape(t2, {4,3,2,3});
    Tensor res2 = reshape(t2, {72});
    Tensor res3 = reshape(t2, {2,1,2,9,2});
    CHECK(check_tensor_equal(res1, Tensor(expected_data, {4,3,2,3})));
    CHECK(check_tensor_equal(res2, Tensor(expected_data, {72})));
    CHECK(check_tensor_equal(res3, Tensor(expected_data, {2,1,2,9,2})));
    CHECK(t2.impl()->storage != res1.impl()->storage);
    CHECK(t2.impl()->storage != res2.impl()->storage);
    CHECK(t2.impl()->storage != res3.impl()->storage);
}

TEST_CASE("Tensor slice") {
    Tensor t1({
            1,2,3,4,
            5,6,7,8,
            9,10,11,12,

            13,14,15,16,
            17,18,19,20,
            21,22,23,24}, {2,3,4});

    Tensor res1 = t1(std::vector<TensorIndex>{0, 0, 0});
    Tensor res2 = t1(std::vector<TensorIndex>{1, 1, 2});
    Tensor res3 = t1(std::vector<TensorIndex>{0, 1});
    Tensor res4 = t1(std::vector<TensorIndex>{1, Slice(0, 2), Slice(0,4,2)});
    CHECK(check_tensor_equal(res1, Tensor(1, {1})));
    CHECK(check_tensor_equal(res2, Tensor(19, {1})));
    CHECK(check_tensor_equal(res3, Tensor({5,6,7,8}, {4})));
    CHECK(check_tensor_equal(res4, Tensor({13,15,17,19}, {2,2})));
}

TEST_CASE("Tensor slice+operations") {
    Tensor t1({
            1,2,3,4,
            5,6,7,8,
            9,10,11,12}, {3,4});
    Tensor t2({
            1,2,
            3,4,
            5,6}, {3,2});
    Tensor res1 = matmul(t1(std::vector<TensorIndex>{Slice{1,3}, Slice{1,4}}), t2);
    Tensor expected1({67,88,103,136}, {2,2});
}
