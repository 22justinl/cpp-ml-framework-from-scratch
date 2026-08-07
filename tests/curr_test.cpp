#include "doctest.h"

#include "core/tensor.h"
#include "data/data.h"
#include "ops/loss.h"
#include "ops/math_ops.h"
#include "ops/tensor_ops.h"
#include "utils/tensor_utils.h"
#include <iostream>

// void print_image(Tensor t) {
//     for (size_t i = 0; i < 28; ++i) {
//         for (size_t j = 0; j < 27; ++j) {
//             std::cout << (t({i, j}) > 0 ? "# " : ". ");
//         }
//         std::cout << t({i, 27}) << std::endl;
//     }
// }

// TEST_CASE("Read IDX file") {
//     Tensor X_test = load_data_idx("./data/MNIST/raw/t10k-images-idx3-ubyte");
//     Tensor Y_test = load_data_idx("./data/MNIST/raw/t10k-labels-idx1-ubyte");
//     Tensor X_train = load_data_idx("./data/MNIST/raw/train-images-idx3-ubyte");
//     Tensor Y_train = load_data_idx("./data/MNIST/raw/train-labels-idx1-ubyte");
//     // X_test = X_test/255;
//     // X_train = X_test/255;
//     print_image(X_test(std::vector<TensorIndex>{0}));
//     Y_test(std::vector<TensorIndex>{0}).print();
//     print_image(X_train(std::vector<TensorIndex>{3}));
//     Y_train(std::vector<TensorIndex>{3}).print();
// }

TEST_CASE("Tensor matrix incrementing") {
    Tensor t1({
            1,2,
            3,4,

            5,6,
            7,8,

            9,10,
            11,12,

            13,14,
            15,16}, {2,2,2,2});
    Tensor t2({
            17,18,19,
            20,21,22,

            23,24,25,
            26,27,28,

            29,30,31,
            32,33,34,

            35,36,37,
            38,39,40}, {2,2,2,3});
    Tensor t3({
            41,42,43,
            44,45,46,

            47,48,49,
            50,51,52,

            53,54,55,
            56,57,58,

            59,60,61,
            62,63,64
            }, {2,2,2,3});
    std::vector<size_t> idx(t1.shape().size()-2, 0);
    size_t t1_offset = 0;
    size_t t2_offset = 0;
    size_t t3_offset = 0;
    std::vector<size_t> t1_matrix_shape = {t1.shape()[2], t1.shape()[3]};
    std::vector<size_t> t1_matrix_strides = {t1.strides_raw()[2], t1.strides_raw()[3]};
    std::vector<size_t> t2_matrix_shape = {t2.shape()[2], t2.shape()[3]};
    std::vector<size_t> t2_matrix_strides = {t2.strides_raw()[2], t2.strides_raw()[3]};
    std::vector<size_t> t3_matrix_shape = {t3.shape()[2], t3.shape()[3]};
    std::vector<size_t> t3_matrix_strides = {t3.strides_raw()[2], t3.strides_raw()[3]};
    for (size_t i = 0; i < 4; ++i) {
        // Tensor(std::make_shared<TensorImpl>(t1.impl()->storage, t1_matrix_shape, t1_matrix_strides, t1_offset, false)).print();
        // Tensor(std::make_shared<TensorImpl>(t2.impl()->storage, t2_matrix_shape, t2_matrix_strides, t2_offset, false)).print();
        // Tensor(std::make_shared<TensorImpl>(t3.impl()->storage, t3_matrix_shape, t3_matrix_strides, t3_offset, false)).print();
        increment_offset_matmul_op(idx, t3.shape(), t1_offset, t1.strides_raw(), t2_offset, t2.strides_raw(), t3_offset, t3.strides_raw());
    }
}

TEST_CASE("Tensor matrix incrementing (non-contiguous)") {
    Tensor t1({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}, {2,2,2,2});
    t1 = transpose(transpose(t1,2,3),0,2);
    // 1,3
    // 9,11,
    //
    // 5,7,
    // 13,15
    //
    // 2,4,
    // 10,12,
    //
    // 6,8,
    // 14,16
    Tensor t2({17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40}, {2,2,2,3});
    t2 = transpose(transpose(t2,0,1),0,2);
    // 17, 18, 19
    // 23, 24, 25
    //
    // 29, 30, 31
    // 35, 36, 37
    //
    // 20, 21, 22
    // 26, 27, 28
    //
    // 32, 33, 34
    // 38, 39, 40
    Tensor t3({41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64}, {2,2,2,3});
    // 41, 42, 43
    // 53, 54, 55
    //
    // 47, 48, 49
    // 59, 60, 61
    //
    // 44, 45, 46
    // 56, 57, 58
    //
    // 50, 51, 52
    // 62, 63, 64
    t3 = transpose(t3,0,2);
    std::vector<size_t> idx(t1.shape().size()-2, 0);
    size_t t1_offset = 0;
    size_t t2_offset = 0;
    size_t t3_offset = 0;
    std::vector<size_t> t1_matrix_shape = {t1.shape()[2], t1.shape()[3]};
    std::vector<size_t> t1_matrix_strides = {t1.strides_raw()[2], t1.strides_raw()[3]};
    std::vector<size_t> t2_matrix_shape = {t2.shape()[2], t2.shape()[3]};
    std::vector<size_t> t2_matrix_strides = {t2.strides_raw()[2], t2.strides_raw()[3]};
    std::vector<size_t> t3_matrix_shape = {t3.shape()[2], t3.shape()[3]};
    std::vector<size_t> t3_matrix_strides = {t3.strides_raw()[2], t3.strides_raw()[3]};
    for (size_t i = 0; i < 4; ++i) {
        // Tensor(std::make_shared<TensorImpl>(t1.impl()->storage, t1_matrix_shape, t1_matrix_strides, t1_offset, false)).print();
        // Tensor(std::make_shared<TensorImpl>(t2.impl()->storage, t2_matrix_shape, t2_matrix_strides, t2_offset, false)).print();
        // Tensor(std::make_shared<TensorImpl>(t3.impl()->storage, t3_matrix_shape, t3_matrix_strides, t3_offset, false)).print();
        increment_offset_matmul_op(idx, t3.shape(), t1_offset, t1.strides_raw(), t2_offset, t2.strides_raw(), t3_offset, t3.strides_raw());
    }
}


TEST_CASE("Tensor nD matmul") {
    Tensor t1({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}, {2,2,2,2});
    Tensor t2({17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40}, {2,2,2,3});
    Tensor t3 = matmul(t1, t2);
    Tensor expected({
            57,60,63,
            131,138,145,

            271,282,293,
            369,384,399,

            581,600,619,
            703,726,749,

            987,1014,1041,
            1133,1164,1195
          }, {2,2,2,3});
    CHECK(check_tensor_equal(t3, expected));
}
TEST_CASE("Tensor nD matmul (non-contiguous)") {
    Tensor t1({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}, {2,2,2,2});
    t1 = transpose(transpose(t1,2,3),0,2);
    Tensor t2({17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40}, {2,2,2,3});
    t2 = transpose(transpose(t2,0,1),0,2);
    Tensor t3 = matmul(t1, t2);
    Tensor expected({
            86,90,94,
            406,426,446,

            390,402,414,
            902,930,958,


            144,150,156,
            512,534,556,

            496,510,524,
            1056,1086,1116}, {2,2,2,3});
    CHECK(check_tensor_equal(t3, expected));
}
TEST_CASE("Tensor nD mmadd") {
    Tensor t1({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}, {2,2,2,2});
    Tensor t2({17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40}, {2,2,2,3});
    Tensor t3({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, {2,2,2,3});
    Tensor t4 = mmadd(t1, t2, t3);
    Tensor expected({
            58,62,66,
            135,143,151,

            278,290,302,
            379,395,411,


            594,614,634,
            719,743,767,

            1006,1034,1062,
            1155,1187,1219}, {2,2,2,3});
    CHECK(check_tensor_equal(t4, expected));
}
TEST_CASE("Tensor nD mmadd (non-contiguous)") {
    Tensor t1({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}, {2,2,2,2});
    t1 = transpose(transpose(t1,2,3),0,2);
    Tensor t2({17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40}, {2,2,2,3});
    t2 = transpose(transpose(t2,0,1),0,2);
    Tensor t3({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, {2,2,2,3});
    t3 = transpose(t3,0,2);
    Tensor t4 = mmadd(t1, t2, t3);
    Tensor expected({
            87,92,97,
            419,440,461,

            397,410,423,
            921,950,979,


            148,155,162,
            528,551,574,

            506,521,536,
            1078,1109,1140
            }, {2,2,2,3});
    CHECK(check_tensor_equal(t4, expected));
}
