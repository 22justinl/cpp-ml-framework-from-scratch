#pragma once

#include "module.h"

namespace nn {
class Flatten: public Module {
public:
    Flatten(size_t start_dim = 1, ssize_t end_dim = -1, std::string name = "Flatten");
    Tensor forward(Tensor x) override;
private:
    size_t start_dim;
    ssize_t end_dim;
};
}
