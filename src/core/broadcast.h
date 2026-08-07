#pragma once

#include "tensor.h"

#include <vector>

struct BroadcastInfo {
    std::vector<size_t> out_shape;
    std::vector<size_t> a_strides;
    std::vector<size_t> b_strides;
    std::vector<size_t> out_strides;
};

struct MatmulBroadcastInfo {
    std::vector<size_t> out_shape;
    std::vector<size_t> a_strides;
    std::vector<size_t> b_strides;
    std::vector<size_t> c_strides;
    std::vector<size_t> out_strides;
};

BroadcastInfo construct_broadcast_info(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b);
MatmulBroadcastInfo construct_matrix_broadcast_info(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b, std::shared_ptr<const TensorImpl> c = nullptr);
const std::vector<size_t> compute_broadcast_shape(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2, size_t ignore_suffix_count = 0);
const std::vector<size_t> compute_broadcast_strides(const std::vector<size_t>& out_shape, const std::vector<size_t>& shape, const std::vector<size_t>& strides, size_t ignore_suffix_count = 0);

std::shared_ptr<TensorImpl> reduce_to_shape(std::shared_ptr<TensorImpl> a, const std::vector<size_t>& shape);
std::shared_ptr<TensorImpl> broadcast_copy(std::shared_ptr<const TensorImpl> a, const std::vector<size_t>& out_shape, const std::vector<size_t>& a_strides, bool requires_grad);
