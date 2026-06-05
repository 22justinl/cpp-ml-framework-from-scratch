#include "sgd.h"
#include "kernels/math_ops.h"

SGD::SGD(std::vector<nn::Parameter*> parameters, float lr) {
    lr_ = lr;
    parameters_ = parameters;
}

void SGD::step() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        nn::Parameter* p_ptr = parameters_[i];
        std::shared_ptr<TensorImpl> g = kernels::scalar_mul(lr_, p_ptr->tensor.grad().impl());
        kernels::sub_inplace(p_ptr->tensor.impl(), g);
    }
}
