#include "activation.h"

#include "kernels/math_ops.h"
#include "kernels/reduction_ops.h"

#include <cmath>

using std::shared_ptr;
using std::make_shared;

namespace kernels {
shared_ptr<TensorImpl> sigmoid(shared_ptr<const TensorImpl> a) {
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(a->storage->data, a->shape, a->strides, a->requires_grad);
    for (size_t i = 0; i < a->storage->data.size(); ++i) {
        if (res->storage->data[i] > 0) {
            res->storage->data[i] = 1.0 / (1.0 + std::exp(-res->storage->data[i]));
        } else {
            float e_x = std::exp(res->storage->data[i]);
            res->storage->data[i] = e_x / (1.0 + e_x);
        }
    }
    return res;
}
shared_ptr<TensorImpl> relu(shared_ptr<const TensorImpl> a) {
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(a->storage->data, a->shape, a->strides, a->requires_grad);
    for (size_t i = 0; i < a->storage->data.size(); ++i) {
        if (res->storage->data[i] < 0) {
            res->storage->data[i] = 0;
        }
    }
    return res;
}
shared_ptr<TensorImpl> softmax(shared_ptr<const TensorImpl> a) {
    // NOTE: Assume col or row tensor
    if (a->shape.size() != 1 && !(a->shape.size() == 2 && (a->shape[0] == 1 || a->shape[1] == 1))) {
        throw std::runtime_error("Softmax only supported for row and column tensors");
    }
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(a->storage->data, a->shape, a->strides, a->requires_grad);
    float max_val = kernels::max(a)->storage->data[0];
    res = kernels::exp(kernels::sub(res, make_shared<TensorImpl>(max_val, a->shape, a->strides, false)));
    float denom = kernels::sum(res)->storage->data[0];
    for (size_t i = 0; i < a->storage->data.size(); ++i) {
        res->storage->data[i] /= denom;
    }
    return res;
}
}
