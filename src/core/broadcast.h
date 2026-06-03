#pragma once

#include "tensor.h"

#include <vector>

struct BroadcastInfo {
    std::vector<size_t> out_shape;
    std::vector<size_t> a_strides;
    std::vector<size_t> b_strides;
    std::vector<size_t> out_strides;
};

BroadcastInfo construct_broadcast_info(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b);
const std::vector<size_t> compute_broadcast_shape(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2);
const std::vector<size_t> compute_broadcast_strides(const std::vector<size_t>& out_shape, const std::vector<size_t>& shape, const std::vector<size_t>& strides);
void map_offset_to_idx(size_t offset, const std::vector<size_t>& strides, std::vector<size_t>& idx); // inplace
size_t map_idx_to_offset(const std::vector<size_t>& idx, const std::vector<size_t>& strides);

std::shared_ptr<TensorImpl> reduce_to_shape(std::shared_ptr<TensorImpl> a, const std::vector<size_t>& shape);
