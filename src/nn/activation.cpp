#include "activation.h"
#include "ops/activation.h"
namespace nn {

ReLU::ReLU(std::string name) {
    set_name(name);
}

Tensor ReLU::forward(Tensor x) {
    return relu(x);
}

Sigmoid::Sigmoid(std::string name) {
    set_name(name);
}

Tensor Sigmoid::forward(Tensor x) {
    return sigmoid(x);
}

Softmax::Softmax(std::string name) {
    set_name(name);
}

Tensor Softmax::forward(Tensor x) {
    return softmax(x);
}

}
