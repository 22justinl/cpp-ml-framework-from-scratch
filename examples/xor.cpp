#include "core/tensor.h"
#include "nn/linear.h"
#include "nn/loss.h"
#include "nn/activation.h"
#include "nn/sequential.h"
#include "optim/sgd.h"
#include "utils/tensor_utils.h"
#include <iostream>

int main() {
    Tensor x[] = {
        Tensor({0,0},{1,2}),
        Tensor({0,1},{1,2}),
        Tensor({1,0},{1,2}),
        Tensor({1,1},{1,2}),
    };
    Tensor y[] = {
        Tensor({0},{1}),
        Tensor({1},{1}),
        Tensor({1},{1}),
        Tensor({0},{1})
    };

    nn::Sequential model({
            std::make_shared<nn::Linear>(2, 4),
            std::make_shared<nn::Sigmoid>(),
            std::make_shared<nn::Linear>(4,1),
            std::make_shared<nn::Sigmoid>()
            });
    nn::MSELoss loss_fn;
    SGD optimizer(model.parameters(), 1e-1);

    size_t epochs = 5000;
    std::cout << "Training:" << std::endl;;
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "[";
        float progress = (float(epoch)/float(epochs));
        size_t count = 70 * progress;
        for (size_t i = 0; i < 70; ++i) {
            if (i <= count) {
                std::cout << "=";
            } else {
                std::cout << " ";
            }
        }
        std::cout << "] " << 100*progress << "%\r";
        std::cout.flush();
        for (size_t i = 0; i < 4; ++i) {
            Tensor pred = model.forward(x[i]);
            Tensor loss = loss_fn.forward(pred, y[i]);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }
    }
    std::cout << std::endl;

    print_tensor(model.forward(Tensor({0,0,0,1,1,0,1,1},{4,2})));
}
