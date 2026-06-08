#include "kernels/reduction_ops.h"
#include "utils/tensor_utils.h"
#include <algorithm>

using std::shared_ptr;
using std::make_shared;

// TODO:nD versions of operations (currently only 0D, 1D, 2D)
namespace kernels {
shared_ptr<TensorImpl> sum(shared_ptr<const TensorImpl> a, size_t dim, bool keepdim) {
    // NOTE: sum using double for better stability (implement more stable algorithm later)
    if (dim == SIZE_T_MAX || (a->shape.size() < 2 && dim == 0)) {
        // sum over all values
        double s = 0;
        for (size_t i = 0; i < a->storage->data.size(); ++i) {
            s += a->storage->data[i];
        }
        shared_ptr<TensorImpl> res = make_shared<TensorImpl>(float(s), std::vector<size_t>({1}), a->requires_grad);
        return res;
    }
    if (dim >= a->shape.size()) {
        throw std::runtime_error("dim argument to sum larger than number of dimensions in tensor");
    }
    if (a->shape.size() > 2) {
        throw std::runtime_error("Sum for tensors with dimensions larger than 2 not supported");
    }
    // 2D tensors
    size_t other_dim = 1-dim;
    std::vector<size_t> new_shape(a->shape);
    if (keepdim) {
        new_shape[dim] = 1;
    } else {
        new_shape.erase(new_shape.begin()+dim);
    }
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, new_shape, a->requires_grad);
    std::vector<size_t> curr_index{0, 0};
    // iterate over other dimension
    for (size_t i = 0; i < a->shape[other_dim]; ++i) {
        curr_index[other_dim] = i;
        double s = 0;
        // sum over dim
        for (size_t j = 0; j < a->shape[dim]; ++j) {
            curr_index[dim] = j;
            s += a->storage->data[calculate_offset(a, curr_index)];
        }
        res->storage->data[i] = float(s);
    }
    return res;
}
shared_ptr<TensorImpl> mean(shared_ptr<const TensorImpl> a, size_t dim, bool keepdim) {
    // TODO: improve stability
    size_t n = dim == SIZE_T_MAX ? a->storage->data.size() : a->shape[dim];
    shared_ptr<TensorImpl> res = sum(a, dim);
    for (size_t i = 0; i < res->storage->data.size(); ++i) {
        res->storage->data[i] /= n;
    }
    return res;
}
shared_ptr<TensorImpl> max(shared_ptr<const TensorImpl> a, size_t dim, bool keepdim, shared_ptr<std::vector<size_t>>* offsets_pptr) {
    if (dim == SIZE_T_MAX || (a->shape.size() < 2 && dim == 0)) {
        // max over all values
        auto it = std::max_element(a->storage->data.begin(), a->storage->data.end());
        shared_ptr<TensorImpl> res = make_shared<TensorImpl>(*it, std::vector<size_t>({1}), a->requires_grad);
        if (offsets_pptr) {
            // Store offsets for gradient calculation
            *offsets_pptr = std::make_unique<std::vector<size_t>>(1);
            (**offsets_pptr)[0] = std::distance(a->storage->data.begin(), it);
        }
        return res;
    }
    if (dim >= a->shape.size()) {
        throw std::runtime_error("dim argument to max larger than number of dimensions in tensor");
    }
    if (a->shape.size() > 2) {
        throw std::runtime_error("Max for tensors with dimensions larger than 2 not supported");
    }
    // 2D tensors
    size_t other_dim = 1-dim;
    std::vector<size_t> new_shape(a->shape);
    if (keepdim) {
        new_shape[dim] = 1;
    } else {
        new_shape.erase(new_shape.begin()+dim);
    }
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, new_shape, a->requires_grad);
    std::vector<size_t> curr_index{0, 0};
    if (offsets_pptr) {
        // Store indices for gradient calculation
        *offsets_pptr = std::make_unique<std::vector<size_t>>(a->shape[other_dim], 0);
    }
    // iterate over other dimension
    for (size_t i = 0; i < a->shape[other_dim]; ++i) {
        curr_index[other_dim] = i;
        curr_index[dim] = 0;
        // setup max value and index
        size_t curr_offset = calculate_offset(a, curr_index);
        size_t v_offset = curr_offset;
        float& v = res->storage->data[i];
        v = a->storage->data[v_offset];
        // find max
        for (size_t j = 0; j < a->shape[dim]; ++j) {
            curr_index[dim] = j;
            curr_offset = calculate_offset(a, curr_index);
            if (a->storage->data[curr_offset] > v) {
                v = a->storage->data[curr_offset];
                v_offset = curr_offset;
            }
        }
        if (offsets_pptr) {
            (**offsets_pptr)[i] = v_offset;
        }
    }
    return res;
}
shared_ptr<TensorImpl> min(shared_ptr<const TensorImpl> a, size_t dim, bool keepdim, shared_ptr<std::vector<size_t>>* offsets_pptr) {
    if (dim == SIZE_T_MAX || (a->shape.size() < 2 && dim == 0)) {
        // min over all values
        auto it = std::min_element(a->storage->data.begin(), a->storage->data.end());
        shared_ptr<TensorImpl> res = make_shared<TensorImpl>(*it, std::vector<size_t>({1}), a->requires_grad);
        if (offsets_pptr) {
            // Store offsets for gradient calculation
            *offsets_pptr = std::make_unique<std::vector<size_t>>(1);
            (**offsets_pptr)[0] = std::distance(a->storage->data.begin(), it);
        }
        return res;
    }
    if (dim >= a->shape.size()) {
        throw std::runtime_error("dim argument to min larger than number of dimensions in tensor");
    }
    if (a->shape.size() > 2) {
        throw std::runtime_error("Min for tensors with dimensions larger than 2 not supported");
    }
    // 2D tensors
    size_t other_dim = 1-dim;
    std::vector<size_t> new_shape(a->shape);
    if (keepdim) {
        new_shape[dim] = 1;
    } else {
        new_shape.erase(new_shape.begin()+dim);
    }
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, new_shape, a->requires_grad);
    std::vector<size_t> curr_index{0, 0};
    if (offsets_pptr) {
        // Store indices for gradient calculation
        *offsets_pptr = std::make_unique<std::vector<size_t>>(a->shape[other_dim], 0);
    }
    // iterate over other dimension
    for (size_t i = 0; i < a->shape[other_dim]; ++i) {
        curr_index[other_dim] = i;
        curr_index[dim] = 0;
        // setup min value and index
        size_t curr_offset = calculate_offset(a, curr_index);
        size_t v_offset = curr_offset;
        float& v = res->storage->data[i];
        v = a->storage->data[v_offset];
        // find min
        for (size_t j = 0; j < a->shape[dim]; ++j) {
            curr_index[dim] = j;
            curr_offset = calculate_offset(a, curr_index);
            if (a->storage->data[curr_offset] < v) {
                v = a->storage->data[curr_offset];
                v_offset = curr_offset;
            }
        }
        if (offsets_pptr) {
            (**offsets_pptr)[i] = v_offset;
        }
    }
    return res;
}
}
