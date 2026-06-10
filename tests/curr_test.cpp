#include "doctest.h"

#include "nn/activation.h"
#include "nn/linear.h"
#include "nn/loss.h"
#include "nn/sequential.h"
#include "optim/sgd.h"
#include <iostream>

TEST_CASE("Test MLP 1") {
    nn::Sequential m({
            std::make_shared<nn::Linear>(3,10),
            std::make_shared<nn::ReLU>(),
            std::make_shared<nn::Linear>(10, 4)
            });
    SGD optimizer(m.parameters());
    nn::MSELoss loss;
    Tensor x1({-1, 1, 0.5}, {1, 3});
    Tensor pred = m.forward(x1);
    Tensor y1({-1,1,2,-2}, {1,4});
    Tensor res1 = loss.forward(pred, y1);
    optimizer.zero_grad();
    res1.backward();
    optimizer.step();
    pred = m.forward(x1);
    Tensor res2 = loss.forward(pred, y1);
    CHECK(res1({0}) > res2({0}));
}
