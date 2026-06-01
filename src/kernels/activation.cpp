#include "activation.h"

#include "kernels/math_ops.h"
#include "kernels/reduction_ops.h"
#include "utils/tensor_utils.h"

#include <cmath>

namespace kernels {
std::shared_ptr<TensorImpl> sigmoid(std::shared_ptr<const TensorImpl> a) {
    std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector(a->data), a->shape, a->strides, a->requires_grad);
    for (size_t i = 0; i < a->data.size(); ++i) {
        if (res->data[i] > 0) {
            res->data[i] = 1.0 / (1.0 + std::exp(-res->data[i]));
        } else {
            float e_x = std::exp(res->data[i]);
            res->data[i] = e_x / (1.0 + e_x);
        }
    }
    return res;
}
std::shared_ptr<TensorImpl> relu(std::shared_ptr<const TensorImpl> a) {
    std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector(a->data), a->shape, a->strides, a->requires_grad);
    for (size_t i = 0; i < a->data.size(); ++i) {
        if (res->data[i] < 0) {
            res->data[i] = 0;
        }
    }
    return res;
}
std::shared_ptr<TensorImpl> softmax(std::shared_ptr<const TensorImpl> a) {
    // NOTE: Assume col or row tensor
    if (a->shape.size() != 1 && !(a->shape.size() == 2 && (a->shape[0] == 1 || a->shape[1] == 1))) {
        throw std::runtime_error("Softmax only supported for row and column tensors");
    }
    std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector(a->data), a->shape, a->strides, a->requires_grad);
    float max_val = kernels::max(a)->data[0];
    res = kernels::exp(kernels::sub(res, create_tensorimpl(std::vector(a->data.size(), max_val), a->shape, a->strides, false)));
    float denom = kernels::sum(res)->data[0];
    for (size_t i = 0; i < a->data.size(); ++i) {
        res->data[i] /= denom;
    }
    return res;
}
}
