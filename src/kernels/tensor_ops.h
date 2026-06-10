#pragma once

#include "core/tensor.h"

namespace kernels {
// transpose, reshape, slicing/view

std::shared_ptr<TensorImpl> transpose(std::shared_ptr<const TensorImpl> a, size_t dim0, size_t dim1);
// std::shared_ptr<TensorImpl> slice(std::shared_ptr<const TensorImpl> a, const std::vector<std::pair<size_t, size_t>> index_ranges);
// std::shared_ptr<TensorImpl> reshape(std::shared_ptr<const TensorImpl> a);
std::shared_ptr<TensorImpl> view(std::shared_ptr<const TensorImpl> a, const std::vector<size_t>& new_shape, const std::vector<size_t>& new_strides, const std::vector<size_t>& offset_idx);
}
