#include "kernels/reduction_ops.h"
#include "utils/tensor_utils.h"
#include <algorithm>

// TODO:nD versions of operations (currently only 0D, 1D, 2D)
namespace kernels {
std::shared_ptr<TensorImpl> sum(std::shared_ptr<const TensorImpl> a, size_t dim) {
    // NOTE: sum using double for better stability (implement more stable algorithm later)
    if (dim == SIZE_T_MAX || (a->shape.size() < 2 && dim == a->shape.size())) {
        // sum over all values
        double s = 0;
        for (size_t i = 0; i < a->data.size(); ++i) {
            s += a->data[i];
        }
        std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector(1, float(s)), {1}, {1}, a->requires_grad);
        return res;
    }
    if (dim > a->shape.size()) {
        throw std::runtime_error("dim argument to reduction larger than number of dimensions in tensor");
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
std::shared_ptr<TensorImpl> max(std::shared_ptr<const TensorImpl> a, size_t dim, std::shared_ptr<std::vector<size_t>>* indices_pptr) {
    if (dim == SIZE_T_MAX || (a->shape.size() < 2 && dim == a->shape.size())) {
        if (indices_pptr) {
            // Store indices for gradient calculation
            *indices_pptr = std::make_unique<std::vector<size_t>>(1);
        }
        // max over all values
        auto it = std::max_element(a->data.begin(), a->data.end());
        (**indices_pptr)[0] = std::distance(a->data.begin(), it);
        float s = *it;
        std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector(1, float(s)), {1}, {1}, a->requires_grad);
        return res;
    }
    if (dim >= a->shape.size()) {
        throw std::runtime_error("dim argument to reduction larger than number of dimensions in tensor");
    }

    // 2D tensors
    if (a->shape.size() > 2) {
        throw std::runtime_error("Max for tensors with dimensions larger than 2 not supported");
    }
    if (indices_pptr) {
        // Store indices for gradient calculation
        *indices_pptr = std::make_unique<std::vector<size_t>>(a->shape[1-dim], 0);
    }
    std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector(a->shape[1-dim], 0.f), {a->shape[1-dim]}, {1}, a->requires_grad);
    std::vector<size_t> curr_index{0, 0};
    for (size_t i = 0; i < a->shape[1-dim]; ++i) {
        // setup indices
        curr_index[1-dim] = i;
        curr_index[dim] = 0;

        // setup max value and index
        size_t idx = 0;
        float& v = res->data[i];
        v = a->data[calculate_offset(a, curr_index)];

        // find max
        for (size_t j = 0; j < a->shape[dim]; ++j) {
            curr_index[dim] = j;
            if (a->data[calculate_offset(a, curr_index)] > v) {
                v = a->data[calculate_offset(a, curr_index)];
                idx = j;
            }
        }

        // set max val and indices
        curr_index[dim] = idx;
        if (indices_pptr) {
            (**indices_pptr)[i] = idx;
        }
    }
    return res;
}
std::shared_ptr<TensorImpl> min(std::shared_ptr<const TensorImpl> a, size_t dim, std::shared_ptr<std::vector<size_t>>* indices_pptr) {
    if (dim == SIZE_T_MAX || (a->shape.size() < 2 && dim == a->shape.size())) {
        if (indices_pptr) {
            // Store indices for gradient calculation
            *indices_pptr = std::make_unique<std::vector<size_t>>(1);
        }
        // min over all values
        auto it = std::min_element(a->data.begin(), a->data.end());
        (**indices_pptr)[0] = std::distance(a->data.begin(), it);
        float s = *it;
        std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector(1, float(s)), {1}, {1}, a->requires_grad);
        return res;
    }
    if (dim >= a->shape.size()) {
        throw std::runtime_error("dim argument to reduction larger than number of dimensions in tensor");
    }

    // 2D tensors
    if (a->shape.size() > 2) {
        throw std::runtime_error("Min for tensors with dimensions larger than 2 not supported");
    }
    if (indices_pptr) {
        // Store indices for gradient calculation
        *indices_pptr = std::make_unique<std::vector<size_t>>(a->shape[1-dim], 0);
    }
    std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector(a->shape[1-dim], 0.f), {a->shape[1-dim]}, {1}, a->requires_grad);
    std::vector<size_t> curr_index{0, 0};
    for (size_t i = 0; i < a->shape[1-dim]; ++i) {
        // setup indices
        curr_index[1-dim] = i;
        curr_index[dim] = 0;

        // setup min value and index
        size_t idx = 0;
        float& v = res->data[i];
        v = a->data[calculate_offset(a, curr_index)];

        // find min
        for (size_t j = 0; j < a->shape[dim]; ++j) {
            curr_index[dim] = j;
            if (a->data[calculate_offset(a, curr_index)] < v) {
                v = a->data[calculate_offset(a, curr_index)];
                idx = j;
            }
        }

        // set min val and indices
        curr_index[dim] = idx;
        if (indices_pptr) {
            (**indices_pptr)[i] = idx;
        }
    }
    return res;
}
}
