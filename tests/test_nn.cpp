#include "doctest.h"
#include "nn/loss.h"
#include "nn/linear.h"
#include "nn/activation.h"
#include "nn/sequential.h"
#include "optim/sgd.h"
#include "utils/tensor_utils.h"

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

TEST_CASE("Test MLP 2") {
    nn::Sequential m({
            std::make_shared<nn::Linear>(3,10),
            std::make_shared<nn::ReLU>(),
            std::make_shared<nn::Linear>(10, 4)
            });
    SGD optimizer(m.parameters(), 1e-1);
    nn::MSELoss loss;

    Tensor x1({-1, 1, 0.5}, {1, 3});
    Tensor pred = m.forward(x1);
    Tensor y1({-1,1,2,-2}, {1,4});
    Tensor res = loss.forward(pred, y1);

    for (size_t i = 0; i < 20; ++i) {
        pred = m.forward(x1);
        res = loss.forward(pred, y1);
        optimizer.zero_grad();
        res.backward();
        optimizer.step();
    }
    // print_tensor(m.forward(x1));
    // print_tensor(y1);
    CHECK(check_tensor_equal(m.forward(x1), y1, 1e-1));
}
