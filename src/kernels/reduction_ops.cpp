#include "kernels/reduction_ops.h"

#include "utils/tensor_utils.h"

#include <algorithm>

using std::shared_ptr;
using std::make_shared;

namespace kernels {
shared_ptr<TensorImpl> sum(shared_ptr<const TensorImpl> a, size_t dim, bool keepdim) {
    // NOTE: sum using double for better stability (implement more stable algorithm later)
    if (dim == SIZE_MAX || (a->shape.size() < 2 && dim == 0)) {
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
    if (keepdim) {
        new_shape[dim] = 1;
    } else {
        new_shape.erase(new_shape.begin()+dim);
    }
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, new_shape, a->requires_grad);
    std::vector<size_t> out_idx(new_shape.size(), 0);
    std::vector<size_t> curr_idx;
    // iterate over each output index
    for (size_t i = 0; i < res->n_el; ++i) {
        double s = 0;
        curr_idx = out_idx;
        if (!keepdim) {
            // input and output indices have different dims, insert 0 before iterating over dim
            curr_idx.insert(curr_idx.begin()+dim, 0);
        }
        // iterate over dim
        for (size_t j = 0; j < a->shape[dim]; ++j) {
            curr_idx[dim] = j;
            s += a->storage->data[idx_to_offset(curr_idx, a->strides, a->offset)];
        }
        res->storage->data[idx_to_offset(out_idx, res->strides, res->offset)] = s;
        increment_idx(res, out_idx);
    }
    return res;
}
shared_ptr<TensorImpl> mean(shared_ptr<const TensorImpl> a, size_t dim, bool keepdim) {
    // TODO: improve stability
    size_t n = dim == SIZE_MAX ? a->storage->data.size() : a->shape[dim];
    shared_ptr<TensorImpl> res = sum(a, dim);
    for (size_t i = 0; i < res->storage->data.size(); ++i) {
        res->storage->data[i] /= n;
    }
    return res;
}

// nD max
shared_ptr<TensorImpl> max_nd(shared_ptr<const TensorImpl> a, size_t dim, bool keepdim) {
    std::vector<size_t> new_shape(a->shape);
    if (keepdim) {
        new_shape[dim] = 1;
    } else {
        new_shape.erase(new_shape.begin()+dim);
    }
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, new_shape, a->requires_grad);
    std::vector<size_t> out_idx(new_shape.size(), 0);
    std::vector<size_t> curr_idx;
    // iterate over each output index
    for (size_t i = 0; i < res->n_el; ++i) {
        curr_idx = out_idx;
        if (!keepdim) {
            // input and output indices have different dims, insert 0 before iterating over dim
            curr_idx.insert(curr_idx.begin()+dim, 0);
        }
        float& v = res->storage->data[idx_to_offset(out_idx, res->strides, res->offset)];
        v = a->storage->data[idx_to_offset(curr_idx, a->strides, a->offset)];
        // iterate over dim
        for (size_t j = 0; j < a->shape[dim]; ++j) {
            v = std::max(v, a->storage->data[idx_to_offset(curr_idx, a->strides, a->offset)]);
        }
        increment_idx(res, out_idx);
    }
    return res;
}
// nD max (store max_pos for gradient calculation)
shared_ptr<TensorImpl> max_nd_store_pos(shared_ptr<const TensorImpl> a, size_t dim, bool keepdim, shared_ptr<TensorImpl>* max_pos_ptr) {
    std::vector<size_t> new_shape(a->shape);
    if (keepdim) {
        new_shape[dim] = 1;
    } else {
        new_shape.erase(new_shape.begin()+dim);
    }
    *max_pos_ptr = std::make_shared<TensorImpl>(0, a->shape, a->strides, false);

    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, new_shape, a->requires_grad);
    std::vector<size_t> out_idx(new_shape.size(), 0);
    std::vector<size_t> curr_idx;
    for (size_t i = 0; i < res->n_el; ++i) {
        curr_idx = out_idx;
        if (!keepdim) {
            curr_idx.insert(curr_idx.begin()+dim, 0);
        }
        float& v = res->storage->data[idx_to_offset(out_idx, res->strides, res->offset)];
        v = a->storage->data[idx_to_offset(curr_idx, a->strides, a->offset)];
        size_t j_v = 0;
        for (size_t j = 0; j < a->shape[dim]; ++j) {
            curr_idx[dim] = j;
            float curr_v = a->storage->data[idx_to_offset(curr_idx, a->strides, a->offset)];
            if (curr_v > v) {
                v = curr_v;
                j_v = j;
            }
        }
        curr_idx[dim] = j_v;
        (*max_pos_ptr)->storage->data[idx_to_offset(curr_idx, a->strides, a->offset)] = 1;
        increment_idx(res, out_idx);
    }
    return res;
}
shared_ptr<TensorImpl> max(shared_ptr<const TensorImpl> a, size_t dim, bool keepdim, shared_ptr<TensorImpl>* max_pos_ptr) {
    if (dim == SIZE_MAX || (a->shape.size() < 2 && dim == 0)) {
        // max over all values
        auto it = std::max_element(a->storage->data.begin(), a->storage->data.end());
        shared_ptr<TensorImpl> res = make_shared<TensorImpl>(*it, std::vector<size_t>((keepdim ? a->shape.size() : 1), 1), a->requires_grad);
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
    if (keepdim) {
        new_shape[dim] = 1;
    } else {
        new_shape.erase(new_shape.begin()+dim);
    }
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, new_shape, a->requires_grad);
    std::vector<size_t> out_idx(new_shape.size(), 0);
    std::vector<size_t> curr_idx;
    // iterate over each output index
    for (size_t i = 0; i < res->n_el; ++i) {
        curr_idx = out_idx;
        if (!keepdim) {
            // input and output indices have different dims, insert 0 before iterating over dim
            curr_idx.insert(curr_idx.begin()+dim, 0);
        }
        float& v = res->storage->data[idx_to_offset(out_idx, res->strides, res->offset)];
        v = a->storage->data[idx_to_offset(curr_idx, a->strides, a->offset)];
        // iterate over dim
        for (size_t j = 0; j < a->shape[dim]; ++j) {
            v = std::min(v, a->storage->data[idx_to_offset(curr_idx, a->strides, a->offset)]);
        }
        increment_idx(res, out_idx);
    }
    return res;
}
// nD min (store min_pos for gradient calculation)
shared_ptr<TensorImpl> min_nd_store_pos(shared_ptr<const TensorImpl> a, size_t dim, bool keepdim, shared_ptr<TensorImpl>* min_pos_ptr) {
    std::vector<size_t> new_shape(a->shape);
    if (keepdim) {
        new_shape[dim] = 1;
    } else {
        new_shape.erase(new_shape.begin()+dim);
    }
    *min_pos_ptr = std::make_shared<TensorImpl>(0, a->shape, a->strides, false);

    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, new_shape, a->requires_grad);
    std::vector<size_t> out_idx(new_shape.size(), 0);
    std::vector<size_t> curr_idx;
    for (size_t i = 0; i < res->n_el; ++i) {
        curr_idx = out_idx;
        if (!keepdim) {
            curr_idx.insert(curr_idx.begin()+dim, 0);
        }
        float& v = res->storage->data[idx_to_offset(out_idx, res->strides, res->offset)];
        v = a->storage->data[idx_to_offset(curr_idx, a->strides, a->offset)];
        size_t j_v = 0;
        for (size_t j = 0; j < a->shape[dim]; ++j) {
            curr_idx[dim] = j;
            float curr_v = a->storage->data[idx_to_offset(curr_idx, a->strides, a->offset)];
            if (curr_v < v) {
                v = curr_v;
                j_v = j;
            }
        }
        curr_idx[dim] = j_v;
        (*min_pos_ptr)->storage->data[idx_to_offset(curr_idx, a->strides, a->offset)] = 1;
        increment_idx(res, out_idx);
    }
    return res;
}
shared_ptr<TensorImpl> min(shared_ptr<const TensorImpl> a, size_t dim, bool keepdim, shared_ptr<TensorImpl>* min_pos_ptr) {
    if (dim == SIZE_MAX || (a->shape.size() < 2 && dim == 0)) {
        // min over all values
        auto it = std::min_element(a->storage->data.begin(), a->storage->data.end());
        shared_ptr<TensorImpl> res = make_shared<TensorImpl>(*it, std::vector<size_t>((keepdim ? a->shape.size() : 1), 1), a->requires_grad);
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
