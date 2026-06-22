#include "core/tensor.h"
#include "nn/linear.h"
#include "nn/loss.h"
#include "nn/activation.h"
#include "nn/sequential.h"
#include "optim/sgd.h"
#include "utils/train_utils.h"

#include <iostream>

int main() {
    Tensor x({
        0,0,
        0,1,
        1,0,
        1,1}, {4,2});
    Tensor y({
        0,
        1,
        1,
        0}, {4,1});

    nn::Sequential model({
            std::make_shared<nn::Linear>(2, 4),
            std::make_shared<nn::Sigmoid>(),
            std::make_shared<nn::Linear>(4,1),
            std::make_shared<nn::Sigmoid>()
            });
    nn::MSELoss loss_fn;
    SGD optimizer(model.parameters(), 1e-1);

    size_t epochs = 5000;
    ProgressBar progress_bar(0, epochs);
    std::cout << "Training:" << std::endl;;
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        progress_bar.increment();
        for (size_t i = 0; i < 4; ++i) {
            Tensor pred = model.forward(x(std::vector<TensorIndex>{Slice{i,i+1}}));
            Tensor loss = loss_fn.forward(pred, y(std::vector<TensorIndex>{Slice{i,i+1}}));
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }
    }

    model.forward(x).print();
}
