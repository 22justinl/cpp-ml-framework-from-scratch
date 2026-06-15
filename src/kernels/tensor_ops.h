#pragma once

#include "core/tensor.h"

namespace kernels {
std::shared_ptr<TensorImpl> transpose(std::shared_ptr<const TensorImpl> a, size_t dim0, size_t dim1);
std::shared_ptr<TensorImpl> squeeze(std::shared_ptr<const TensorImpl> a, size_t dim);
std::shared_ptr<TensorImpl> unsqueeze(std::shared_ptr<const TensorImpl> a, size_t dim);
std::shared_ptr<TensorImpl> reshape(std::shared_ptr<const TensorImpl> a, const std::vector<size_t>& new_shape);
std::shared_ptr<TensorImpl> slice(std::shared_ptr<const TensorImpl> a, const std::vector<TensorIndex>& indices);
}
