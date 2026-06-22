#pragma once

#include "nn/module.h"

namespace nn {
class ReLU: public Module {
public:
    ReLU(std::string name = "ReLU");
    Tensor forward(Tensor x) override;
};
class Sigmoid: public Module {
public:
    Sigmoid(std::string name = "Sigmoid");
    Tensor forward(Tensor x) override;
};
class Softmax: public Module {
public:
    Softmax(std::string name = "Softmax");
    Tensor forward(Tensor x) override;
};
}
