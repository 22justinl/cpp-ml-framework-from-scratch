#include "broadcast.h"

#include "kernels/reduction_ops.h"
#include "utils/tensor_utils.h"

#include <algorithm>
#include <string>

BroadcastInfo construct_broadcast_info(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b) {
    std::vector<size_t> out_shape = compute_broadcast_shape(a->shape, b->shape);
    std::vector<size_t> a_strides = compute_broadcast_strides(out_shape, a->shape, a->strides);
    std::vector<size_t> b_strides = compute_broadcast_strides(out_shape, b->shape, b->strides);
    std::vector<size_t> out_strides = calculate_strides(out_shape);
    return BroadcastInfo{out_shape, a_strides, b_strides, out_strides};
}

MatmulBroadcastInfo construct_matrix_broadcast_info(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b, std::shared_ptr<const TensorImpl> c) {
    std::vector<size_t> out_shape = compute_broadcast_shape(a->shape, b->shape, 2);
    if (c) {
        out_shape = compute_broadcast_shape(c->shape, out_shape, 2);
    }
    out_shape[out_shape.size()-2] = a->shape[a->shape.size()-2];
    out_shape[out_shape.size()-1] = b->shape[b->shape.size()-1];
    std::vector<size_t> a_strides = compute_broadcast_strides(out_shape, a->shape, a->strides, 2);
    std::vector<size_t> b_strides = compute_broadcast_strides(out_shape, b->shape, b->strides, 2);
    std::vector<size_t> c_strides;
    std::vector<size_t> out_strides = calculate_strides(out_shape);
    if (c) {
        c_strides = compute_broadcast_strides(out_shape, c->shape, c->strides, 2);
    } else {
        c_strides = out_strides;
    }
    return MatmulBroadcastInfo{out_shape, a_strides, b_strides, c_strides, out_strides};
}

const std::vector<size_t> compute_broadcast_shape(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2, size_t ignore_suffix_count) {
    size_t min_dims = std::min(shape1.size(), shape2.size());
    size_t max_dims = std::max(shape1.size(), shape2.size());
    std::vector<size_t> out_shape(shape1.size() > shape2.size() ? shape1 : shape2);
    for (size_t i = ignore_suffix_count; i < min_dims; ++i) {
        size_t d1 = shape1[shape1.size()-i-1];
        size_t d2 = shape2[shape2.size()-i-1];
        if (d1 == 0 || d2 == 0 || !(d1 == d2 || d1 == 1 || d2 == 1)) {
            if (!ignore_suffix_count) {
                throw std::runtime_error("Cannot broadcast shapes " + shape_to_string(shape1) + " and " + shape_to_string(shape2));
            } else {
                throw std::runtime_error("Cannot broadcast shapes " + shape_to_string(shape1) + " and " + shape_to_string(shape2) + " ignoring last " + std::to_string(ignore_suffix_count) + " dims");
            }
        }
        out_shape[max_dims-i-1] = std::max(d1, d2);
    }
    return out_shape;
}

const std::vector<size_t> compute_broadcast_strides(const std::vector<size_t>& out_shape, const std::vector<size_t>& shape, const std::vector<size_t>& strides, size_t ignore_suffix_count) {
    size_t diff = out_shape.size() - shape.size();
    size_t broadcast_dims = shape.size() - ignore_suffix_count;
    std::vector<size_t> new_strides(out_shape.size(), 0);
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i < broadcast_dims && shape[i] == 1) {
            continue;
        }
        new_strides[diff + i] = strides[i];
    }
    return new_strides;
}

std::shared_ptr<TensorImpl> reduce_to_shape(std::shared_ptr<TensorImpl> a, const std::vector<size_t>& shape) {
    if (check_shape_match(a->shape, shape)) {
        return a;
    }

    // reduce prepended axes
    while (a->shape.size() != shape.size()) {
        a = kernels::sum(a, 0);
    }
    // reduce trailing axes
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == 1 && a->shape[i] > 1) {
            a = kernels::sum(a, i, true);
        }
    }

    return a;
}

std::shared_ptr<TensorImpl> broadcast_copy(std::shared_ptr<const TensorImpl> a, const std::vector<size_t>& out_shape, const std::vector<size_t>& a_strides, bool requires_grad) {
    std::shared_ptr<TensorImpl> res = std::make_shared<TensorImpl>(0, out_shape, requires_grad);
    std::vector<size_t> idx(out_shape.size(), 0);
    size_t a_offset = a->offset;
    size_t res_offset = res->offset;
    for (size_t i = 0; i < calculate_n_el(out_shape); ++i) {
        res->storage->data[res_offset] = a->storage->data[a_offset];
        increment_offset_unary_op(idx, out_shape, a_offset, a_strides, res_offset, res->strides);
    }
    return res;
}
