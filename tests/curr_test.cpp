#include "doctest.h"

#include "core/tensor.h"
#include "data/data.h"
#include "ops/loss.h"
#include "ops/math_ops.h"
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
//

TEST_CASE("Tensor matmul") {
    Tensor t1({0,1,2,3,4,5,6,7,8,9,10,11}, {3,4});
    Tensor t2({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}, {4,5});
    // Tensor t3({1,1,1}, {1,3});
    t1.print();
    t2.print();
    // t3.print();
    Tensor res1 = matmul(t1,t2);
    // Tensor res2 = matmul(t3,t1);
    Tensor expected1({70,76,82,88,94,190,212,234,256,278,310,348,386,424,462}, {3,5});
    Tensor expected2({12,15,18,21}, {1,4});
    CHECK(check_tensor_equal(res1, expected1));
    // CHECK(check_tensor_equal(res2, expected2));
    res1.print();
    // res2.print();
}
