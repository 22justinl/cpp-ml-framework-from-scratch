#pragma once

#include "module.h"
namespace nn {
class MSELoss: public nn::Module{
public:
    MSELoss(std::string name = "MSELoss");
    Tensor forward(Tensor pred, Tensor target);
};

class CrossEntropyLoss: public nn::Module{
public:
    CrossEntropyLoss(std::string name = "CrossEntropyLoss");
    Tensor forward(Tensor pred, Tensor target);
};
}
