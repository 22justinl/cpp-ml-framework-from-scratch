#include "doctest.h"
#include "nn/loss.h"
#include "nn/linear.h"
#include "nn/activation.h"
#include "nn/parameter.h"
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

TEST_CASE("Test MLP with separate tensors vs slices") {
    std::vector<Tensor> x_list{
        Tensor({0,0}, {1,2}),
        Tensor({0,1}, {1,2}),
        Tensor({1,0}, {1,2}),
        Tensor({1,1}, {1,2}),
    };
    std::vector<Tensor> y_list{
        Tensor({0}, {1,1}),
        Tensor({1}, {1,1}),
        Tensor({1}, {1,1}),
        Tensor({0}, {1,1}),
    };

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


    nn::Sequential model1({
            std::make_shared<nn::Linear>(2, 4),
            std::make_shared<nn::Sigmoid>(),
            std::make_shared<nn::Linear>(4,1),
            std::make_shared<nn::Sigmoid>()
            });
    nn::Sequential model2({
            std::make_shared<nn::Linear>(2, 4),
            std::make_shared<nn::Sigmoid>(),
            std::make_shared<nn::Linear>(4,1),
            std::make_shared<nn::Sigmoid>()
            });
    std::vector<nn::Parameter*> params1 = model1.parameters();
    std::vector<nn::Parameter*> params2 = model2.parameters();
    for (size_t i = 0; i < params1.size(); ++i) {
        params2[i]->tensor = params1[i]->tensor.copy();
    }

    nn::MSELoss loss_fn;
    SGD optimizer1(model1.parameters(), 1e-1);
    SGD optimizer2(model2.parameters(), 1e-1);

    Tensor res1 = model1.forward(x_list[1]);
    Tensor loss1 = loss_fn.forward(res1, y_list[1]);
    Tensor res2 = model2.forward(x(std::vector<TensorIndex>{Slice{1,2}}));
    Tensor loss2 = loss_fn.forward(res2, y(std::vector<TensorIndex>{Slice{1,2}}));

    loss1.backward();
    loss2.backward();

    for (size_t i = 0; i < params1.size(); ++i) {
        CHECK(check_tensor_equal(params2[i]->tensor.grad(), params1[i]->tensor.grad()));
    }
    optimizer1.step();
    optimizer2.step();
    for (size_t i = 0; i < params1.size(); ++i) {
        CHECK(check_tensor_equal(params2[i]->tensor, params1[i]->tensor));
    }

    size_t epochs = 10;
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        optimizer1.zero_grad();
        optimizer2.zero_grad();
        res1 = model1.forward(x_list[1]);
        loss1 = loss_fn.forward(res1, y_list[1]);
        loss1.backward();
        res2 = model2.forward(x(std::vector<TensorIndex>{Slice{1,2}}));
        loss2 = loss_fn.forward(res2, y(std::vector<TensorIndex>{Slice{1,2}}));
        loss2.backward();
        CHECK(check_tensor_equal(res1, res2));

        for (size_t i = 0; i < params1.size(); ++i) {
            CHECK(check_tensor_equal(params2[i]->tensor.grad(), params1[i]->tensor.grad()));
        }
        optimizer1.step();
        optimizer2.step();
        for (size_t i = 0; i < params1.size(); ++i) {
            CHECK(check_tensor_equal(params2[i]->tensor, params1[i]->tensor));
        }
    }
}
