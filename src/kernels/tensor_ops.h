#pragma once

#include "core/tensor.h"
#include <utility>

namespace kernels {
// transpose, reshape, slicing/view

std::shared_ptr<TensorImpl> transpose(std::shared_ptr<const TensorImpl> a);
// std::shared_ptr<TensorImpl> slice(std::shared_ptr<const TensorImpl> a, const std::vector<std::pair<size_t, size_t>> index_ranges);
// std::shared_ptr<TensorImpl> reshape(std::shared_ptr<const TensorImpl> a);
}
