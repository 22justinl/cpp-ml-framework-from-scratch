#pragma once

#include "nn/module.h"
namespace nn {
class Sequential: public Module {
public:
    Sequential(std::initializer_list<std::shared_ptr<Module>> sequence, std::string name = "Sequential");
    Tensor forward(Tensor x) override;
};
}
