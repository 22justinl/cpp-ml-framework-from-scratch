#include "core/tensor.h"
#include "data/data.h"
#include "nn/flatten.h"
#include "nn/linear.h"
#include "nn/loss.h"
#include "nn/activation.h"
#include "nn/sequential.h"
#include "ops/reduction_ops.h"
#include "optim/sgd.h"
#include "utils/train_utils.h"
#include <mach-o/dyld.h>

#include <iostream>
#include <filesystem>

int main() {
    uint32_t buf_size = PATH_MAX;
    std::vector<char> buf(buf_size);
    if (_NSGetExecutablePath(buf.data(), &buf_size) != 0) {
        buf.resize(buf_size);
        if (_NSGetExecutablePath(buf.data(), &buf_size) != 0) {
            throw std::runtime_error("Couldn't get executable path");
        }
    }
    std::filesystem::path executable_path(buf.begin(), buf.end());
    Tensor X_test = load_data_idx(executable_path.parent_path() / "../data/MNIST/raw/t10k-images-idx3-ubyte");
    Tensor Y_test = load_data_idx(executable_path.parent_path() / "../data/MNIST/raw/t10k-labels-idx1-ubyte");
    Tensor X_train = load_data_idx(executable_path.parent_path() / "../data/MNIST/raw/train-images-idx3-ubyte");
    Tensor Y_train = load_data_idx(executable_path.parent_path() / "../data/MNIST/raw/train-labels-idx1-ubyte");

    nn::Sequential model({
            std::make_shared<nn::Flatten>(),
            std::make_shared<nn::Linear>(784, 100),
            std::make_shared<nn::ReLU>(),
            std::make_shared<nn::Linear>(100,100),
            std::make_shared<nn::Sigmoid>(),
            std::make_shared<nn::Linear>(100,10),
            });

    size_t epochs = 10;
    size_t batch_size = 100;
    float lr = 1e-2;

    SGD optimizer(model.parameters(), lr);
    nn::CrossEntropyLoss loss_fn;

    std::cout << "Training:" << std::endl;;
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        // ProgressBar progress_bar(0, X_train.shape()[0] / batch_size);
        for (size_t i = 0; i < X_train.shape()[0]; i+=batch_size) {
            // progress_bar.increment();
            Tensor pred = model.forward(X_train(std::vector<TensorIndex>{Slice{i,i+batch_size}}));
            Tensor loss = loss_fn.forward(pred, Y_train(std::vector<TensorIndex>{Slice{i,i+batch_size}}));
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }
        Tensor test_pred = model.forward(X_test);
        Tensor test_pred_max = max(test_pred, 1);
        size_t test_correct_count = 0;
        for (size_t i = 0; i < X_test.shape()[0]; ++i) {
            if (test_pred_max({i}) == test_pred({i, static_cast<size_t>(Y_test({i}))})) {
                ++test_correct_count;
            }
        }
        float test_acc = static_cast<float>(test_correct_count)/static_cast<float>(X_test.shape()[0]);
        std::cout << "Epoch " << epoch+1 << " Test accuracy: " << test_acc << "%" << "(" << test_correct_count << "/" << X_test.shape()[0] << ")" << std::endl;
        // std::cout << "Epoch " << epoch+1 << std::endl;
    }
    Tensor train_pred = model.forward(X_train);
    Tensor train_pred_max = max(train_pred, 1);
    Tensor test_pred = model.forward(X_test);
    Tensor test_pred_max = max(test_pred, 1);
    size_t train_correct_count = 0;
    size_t test_correct_count = 0;
    for (size_t i = 0; i < X_train.shape()[0]; ++i) {
        if (train_pred_max({i}) == train_pred({i, static_cast<size_t>(Y_train({i}))})) {
            ++train_correct_count;
        }
    }
    for (size_t i = 0; i < X_test.shape()[0]; ++i) {
        if (test_pred_max({i}) == test_pred({i, static_cast<size_t>(Y_test({i}))})) {
            ++test_correct_count;
        }
    }
    float train_acc = static_cast<float>(train_correct_count)/static_cast<float>(X_train.shape()[0]);
    float test_acc = static_cast<float>(test_correct_count)/static_cast<float>(X_test.shape()[0]);

    std::cout << "Train accuracy: " << train_acc << "%" << "(" << train_correct_count << "/" << X_train.shape()[0] << ")" << std::endl;
    std::cout << "Test accuracy: " << test_acc << "%" << "(" << test_correct_count << "/" << X_test.shape()[0] << ")" << std::endl;
}
