#include "broadcast.h"

#include "kernels/reduction_ops.h"
#include "utils/tensor_utils.h"

#include <algorithm>

BroadcastInfo construct_broadcast_info(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b) {
    std::vector<size_t> out_shape = compute_broadcast_shape(a->shape, b->shape);
    std::vector<size_t> a_strides = compute_broadcast_strides(out_shape, a->shape, a->strides);
    std::vector<size_t> b_strides = compute_broadcast_strides(out_shape, b->shape, b->strides);
    std::vector<size_t> out_strides = calculate_strides(out_shape);
    return BroadcastInfo{out_shape, a_strides, b_strides, out_strides};
}

const std::vector<size_t> compute_broadcast_shape(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2) {
    size_t min_dims = std::min(shape1.size(), shape2.size());
    size_t max_dims = std::max(shape1.size(), shape2.size());
    std::vector<size_t> out_shape(shape1.size() > shape2.size() ? shape1 : shape2);
    for (size_t i = 0; i < min_dims; ++i) {
        size_t d1 = shape1[shape1.size()-i-1];
        size_t d2 = shape2[shape2.size()-i-1];
        if (d1 == 0 || d2 == 0 || !(d1 == d2 || d1 == 1 || d2 == 1)) {
            throw std::runtime_error("Cannot broadcast shapes " + shape_to_string(shape1) + " and " + shape_to_string(shape2));
        }
        out_shape[max_dims-i-1] = std::max(d1, d2);
    }
    return out_shape;
}

const std::vector<size_t> compute_broadcast_strides(const std::vector<size_t>& out_shape, const std::vector<size_t>& shape, const std::vector<size_t>& strides) {
    size_t diff = out_shape.size() - shape.size();
    std::vector<size_t> new_strides(out_shape.size(), 0);
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == 1) {
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
