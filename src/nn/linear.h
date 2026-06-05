#pragma once

#include "nn/module.h"

namespace nn {
class Linear: public Module{
public:
    Linear(size_t in_features, size_t out_features, bool bias = true, std::string name = "Linear");
    Tensor forward(Tensor x) override;
private:
    Parameter weight_;
    Parameter bias_;
};
}
