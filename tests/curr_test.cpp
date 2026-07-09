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
