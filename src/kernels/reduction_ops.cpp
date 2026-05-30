#include "kernels/reduction_ops.h"
#include "utils/tensor_utils.h"

namespace kernels {
std::shared_ptr<TensorImpl> sum(std::shared_ptr<const TensorImpl> a, size_t dim) {
    // NOTE: sum using double for better stability (implement more stable algorithm later)
    if (dim == SIZE_T_MAX) {
        // sum over all values
        double s = 0;
        for (size_t i = 0; i < a->data.size(); ++i) {
            s += a->data[i];
        }
        std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector(1, float(s)), {1}, {1}, a->requires_grad);
        return res;
    }
    // TODO:nD sum
    if (dim > a->shape.size()) {
        throw std::runtime_error("dim argument to reduction larger than number of dimensions in tensor");
    }

    // 0D and 1D tensors
    if (a->shape.size() < 2) {
        // sum over all values
        double s = 0;
        for (size_t i = 0; i < a->data.size(); ++i) {
            s += a->data[i];
        }
        std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector(1, float(s)), {1}, {1}, a->requires_grad);
        return res;
    }
    // 2D tensors
    std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector(a->shape[1-dim], 0.f), {a->shape[1-dim]}, {1}, a->requires_grad);
    std::vector<size_t> curr_index{0, 0};

    for (size_t i = 0; i < a->shape[1-dim]; ++i) {
        curr_index[1-dim] = i;
        double s = 0;
        for (size_t j = 0; j < a->shape[dim]; ++j) {
            curr_index[dim] = j;
            s += a->data[calculate_offset(a, curr_index)];
        }
        res->data[i] = float(s);
    }

    return res;
}
std::shared_ptr<TensorImpl> mean(std::shared_ptr<const TensorImpl> a, size_t dim) {
    // TODO: improve stability
    size_t n = dim == SIZE_T_MAX ? a->data.size() : a->shape[dim];
    std::shared_ptr<TensorImpl> res = sum(a, dim);
    for (size_t i = 0; i < res->data.size(); ++i) {
        res->data[i] /= n;
    }
    return res;
}
// std::shared_ptr<TensorImpl> max(std::shared_ptr<const TensorImpl> a, size_t dim) {
//
// }
// std::shared_ptr<TensorImpl> min(std::shared_ptr<const TensorImpl> a, size_t dim) {
//
// }
}
