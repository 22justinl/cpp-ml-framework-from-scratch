#include "relu.h"
#include "ops/activation.h"
namespace nn {

ReLU::ReLU(std::string name) {
    set_name(name);
}

Tensor ReLU::forward(Tensor x) {
    return relu(x);
}
}
