#include "activation.h"

#include "core/broadcast.h"
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
shared_ptr<TensorImpl> softmax(shared_ptr<const TensorImpl> a, size_t dim) {
    if (dim == SIZE_MAX) {
        throw std::runtime_error("Softmax requires dimension argument");
    }
    shared_ptr<TensorImpl> max = kernels::max(a);
    BroadcastInfo b_info = construct_broadcast_info(a, max);
    shared_ptr<TensorImpl> res = kernels::exp(kernels::sub_broadcast(a, max, b_info)); // exp(x - max(x))
    shared_ptr<TensorImpl> denom = kernels::sum(res, dim, true); // sum_i(exp(x_i - max(x)))
    b_info = construct_broadcast_info(res, denom);
    res = kernels::div_broadcast(res, denom, b_info); // exp(x-max(x))/sum_i(exp(x_i - max(x)))
    return res;
}
}
