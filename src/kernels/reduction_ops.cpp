#include "kernels/reduction_ops.h"

#include "kernels/tensor_ops.h"
#include "utils/tensor_utils.h"

#include <algorithm>

using std::shared_ptr;
using std::make_shared;

namespace kernels {
shared_ptr<TensorImpl> sum(shared_ptr<const TensorImpl> a, size_t dim, bool keepdim) {
    // NOTE: sum using double for better stability (implement more stable algorithm later)
    if (dim == SIZE_MAX || (a->shape.size() < 2 && dim == 0)) {
        if (dim == SIZE_MAX && keepdim) { throw std::runtime_error("keepdim cannot be true for sum without specified dim"); }
        // sum over all values
        double s = 0;
        for (size_t i = 0; i < a->storage->data.size(); ++i) {
            s += a->storage->data[i];
        }
        shared_ptr<TensorImpl> res = make_shared<TensorImpl>(float(s), std::vector<size_t>((keepdim ? a->shape.size() : 1), 1), a->requires_grad);
        return res;
    }

    if (dim >= a->shape.size()) {
        throw std::runtime_error("dim argument to sum larger than number of dimensions in tensor");
    }
    std::vector<size_t> new_shape(a->shape);
    std::vector<size_t> iter_shape(a->shape);
    new_shape[dim] = 1;
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, new_shape, a->requires_grad);
    std::vector<size_t> idx(iter_shape.size(), 0);
    const std::vector<size_t>& a_shape = a->shape;

    size_t a_offset = a->offset;
    size_t res_offset = res->offset;

    const std::vector<size_t>& a_strides = a->strides;
    const std::vector<size_t>& res_strides = res->strides;

    const std::vector<float>& a_data = a->storage->data;
    std::vector<float>& res_data = res->storage->data;

    const size_t n_el = res->n_el;
    for (size_t i = 0; i < n_el; ++i) {
        double s = 0;
        size_t a_sum_offset = a_offset;
        for (size_t j = 0; j < a_shape[dim]; ++j) {
            s += a_data[a_sum_offset];
            a_sum_offset += a_strides[dim];
        }
        res_data[res_offset] = float(s);
        increment_offset_unary_op(idx, new_shape, a_offset, a_strides, res_offset, res_strides);
    }

    if (!keepdim) {
        res = kernels::squeeze(res, dim);
    }
    return res;
}
shared_ptr<TensorImpl> mean(shared_ptr<const TensorImpl> a, size_t dim, bool keepdim) {
    // TODO: improve stability
    if (dim == SIZE_MAX && keepdim) { throw std::runtime_error("keepdim cannot be true for mean without specified dim"); }
    size_t n = dim == SIZE_MAX ? a->storage->data.size() : a->shape[dim];
    shared_ptr<TensorImpl> res = sum(a, dim, keepdim);
    std::vector<float>& res_data = res->storage->data;
    for (size_t i = 0; i < res_data.size(); ++i) {
        res_data[i] /= n;
    }
    return res;
}

// nD max
shared_ptr<TensorImpl> max_nd(shared_ptr<const TensorImpl> a, size_t dim, bool keepdim) {
    std::vector<size_t> new_shape(a->shape);
    new_shape[dim] = 1;

    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, new_shape, a->requires_grad);

    std::vector<size_t> idx(new_shape.size());
    const std::vector<size_t>& a_shape = a->shape;

    size_t a_offset = a->offset;
    size_t res_offset = res->offset;

    const std::vector<size_t>& a_strides = a->strides;
    const std::vector<size_t>& res_strides = res->strides;

    const std::vector<float>& a_data = a->storage->data;
    std::vector<float>& res_data = res->storage->data;

    const size_t n_el = res->n_el;
    for (size_t i = 0; i < n_el; ++i) {
        float& v = res_data[res_offset];
        size_t a_max_offset = a_offset;
        for (size_t j = 0; j < a_shape[dim]; ++j) {
            v = std::max(v, a_data[a_max_offset]);
            a_max_offset += a_strides[dim];
        }
        increment_offset_unary_op(idx, new_shape, a_offset, a_strides, res_offset, res_strides);
    }

    if (!keepdim) {
        res = kernels::squeeze(res, dim);
    }

    return res;
}
// nD max (store max_pos for gradient calculation)
shared_ptr<TensorImpl> max_nd_store_pos(shared_ptr<const TensorImpl> a, size_t dim, bool keepdim, shared_ptr<TensorImpl>* max_pos_ptr) {
    std::vector<size_t> new_shape(a->shape);
    new_shape[dim] = 1;
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, new_shape, a->requires_grad);
    shared_ptr<TensorImpl> max_pos = std::make_shared<TensorImpl>(0, a->shape, a->strides, false);

    std::vector<size_t> idx(new_shape.size(), 0);
    const std::vector<size_t>& a_shape = a->shape;

    size_t a_offset = a->offset;
    size_t res_offset = res->offset;

    const std::vector<size_t>& a_strides = a->strides;
    const std::vector<size_t>& res_strides = res->strides;

    const std::vector<float>& a_data = a->storage->data;
    std::vector<float>& res_data = res->storage->data;
    std::vector<float>& max_pos_data = max_pos->storage->data;

    const size_t n_el = res->n_el;
    for (size_t i = 0; i < n_el; ++i) {
        float curr_val = a_data[a_offset];
        float& v = res_data[res_offset];
        size_t j_v = 0;
        v = curr_val;

        size_t a_max_offset = a_offset;
        for (size_t j = 0; j < a_shape[dim]; ++j) {
            curr_val = a_data[a_max_offset];
            if (curr_val > v) {
                v = curr_val;
                j_v = j;
            }
            a_max_offset += a_strides[dim];
        }
        max_pos_data[a_offset+j_v*a_strides[dim]] = 1;
        increment_offset_unary_op(idx, new_shape, a_offset, a_strides, res_offset, res_strides);
    }

    if (!keepdim) {
        res = kernels::squeeze(res, dim);
    }
    *max_pos_ptr = max_pos;
    return res;
}
shared_ptr<TensorImpl> max(shared_ptr<const TensorImpl> a, size_t dim, bool keepdim, shared_ptr<TensorImpl>* max_pos_ptr) {
    if (dim == SIZE_MAX || (a->shape.size() < 2 && dim == 0)) {
        if (dim == SIZE_MAX && keepdim) { throw std::runtime_error("keepdim cannot be true for max without specified dim"); }
        // max over all values
        auto it = std::max_element(a->storage->data.begin(), a->storage->data.end());
        shared_ptr<TensorImpl> res = make_shared<TensorImpl>(*it, std::vector<size_t>{1}, a->requires_grad);
        if (max_pos_ptr) {
            // Store max positions for gradient calculation
            *max_pos_ptr = std::make_shared<TensorImpl>(0, a->shape, a->strides, false);
            (*max_pos_ptr)->storage->data[std::distance(a->storage->data.begin(), it)] = 1;
        }
        return res;
    }
    if (dim >= a->shape.size()) {
        throw std::runtime_error("dim argument to max larger than number of dimensions in tensor");
    }
    if (max_pos_ptr) {
        return max_nd_store_pos(a, dim, keepdim, max_pos_ptr);
    }
    return max_nd(a, dim, keepdim);
}

// nD min
shared_ptr<TensorImpl> min_nd(shared_ptr<const TensorImpl> a, size_t dim, bool keepdim) {
    std::vector<size_t> new_shape(a->shape);
    new_shape[dim] = 1;

    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, new_shape, a->requires_grad);

    std::vector<size_t> idx(new_shape.size());
    const std::vector<size_t>& a_shape = a->shape;

    size_t a_offset = a->offset;
    size_t res_offset = res->offset;

    const std::vector<size_t>& a_strides = a->strides;
    const std::vector<size_t>& res_strides = res->strides;

    const std::vector<float>& a_data = a->storage->data;
    std::vector<float>& res_data = res->storage->data;

    const size_t n_el = res->n_el;
    for (size_t i = 0; i < n_el; ++i) {
        float& v = res_data[res_offset];
        size_t a_min_offset = a_offset;
        for (size_t j = 0; j < a_shape[dim]; ++j) {
            v = std::min(v, a_data[a_min_offset]);
            a_min_offset += a_strides[dim];
        }
        increment_offset_unary_op(idx, new_shape, a_offset, a_strides, res_offset, res_strides);
    }

    if (!keepdim) {
        res = kernels::squeeze(res, dim);
    }

    return res;
}
// nD min (store min_pos for gradient calculation)
shared_ptr<TensorImpl> min_nd_store_pos(shared_ptr<const TensorImpl> a, size_t dim, bool keepdim, shared_ptr<TensorImpl>* min_pos_ptr) {
    std::vector<size_t> new_shape(a->shape);
    new_shape[dim] = 1;
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, new_shape, a->requires_grad);
    shared_ptr<TensorImpl> min_pos = std::make_shared<TensorImpl>(0, a->shape, a->strides, false);

    std::vector<size_t> idx(new_shape.size(), 0);
    const std::vector<size_t>& a_shape = a->shape;

    size_t a_offset = a->offset;
    size_t res_offset = res->offset;

    const std::vector<size_t>& a_strides = a->strides;
    const std::vector<size_t>& res_strides = res->strides;

    const std::vector<float>& a_data = a->storage->data;
    std::vector<float>& res_data = res->storage->data;
    std::vector<float>& min_pos_data = min_pos->storage->data;

    const size_t n_el = res->n_el;
    for (size_t i = 0; i < n_el; ++i) {
        float curr_val = a_data[a_offset];
        float& v = res_data[res_offset];
        size_t j_v = 0;
        v = curr_val;

        size_t a_min_offset = a_offset;
        for (size_t j = 0; j < a_shape[dim]; ++j) {
            curr_val = a_data[a_min_offset];
            if (curr_val < v) {
                v = curr_val;
                j_v = j;
            }
            a_min_offset += a_strides[dim];
        }
        min_pos_data[a_offset+j_v*a_strides[dim]] = 1;
        increment_offset_unary_op(idx, new_shape, a_offset, a_strides, res_offset, res_strides);
    }

    if (!keepdim) {
        res = kernels::squeeze(res, dim);
    }
    *min_pos_ptr = min_pos;
    return res;
}
shared_ptr<TensorImpl> min(shared_ptr<const TensorImpl> a, size_t dim, bool keepdim, shared_ptr<TensorImpl>* min_pos_ptr) {
    if (dim == SIZE_MAX || (a->shape.size() < 2 && dim == 0)) {
        if (dim == SIZE_MAX && keepdim) { throw std::runtime_error("keepdim cannot be true for min without specified dim"); }
        // min over all values
        auto it = std::min_element(a->storage->data.begin(), a->storage->data.end());
        shared_ptr<TensorImpl> res = make_shared<TensorImpl>(*it, std::vector<size_t>{1}, a->requires_grad);
        if (min_pos_ptr) {
            // Store min positions for gradient calculation
            *min_pos_ptr = std::make_shared<TensorImpl>(0, a->shape, a->strides, false);
            (*min_pos_ptr)->storage->data[std::distance(a->storage->data.begin(), it)] = 1;
        }
        return res;
    }
    if (dim >= a->shape.size()) {
        throw std::runtime_error("dim argument to min larger than number of dimensions in tensor");
    }
    if (min_pos_ptr) {
        return min_nd_store_pos(a, dim, keepdim, min_pos_ptr);
    }
    return min_nd(a, dim, keepdim);
}
}
