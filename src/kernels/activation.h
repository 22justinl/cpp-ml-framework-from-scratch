#pragma once

#include "core/tensor.h"

namespace kernels {
std::shared_ptr<TensorImpl> sigmoid(std::shared_ptr<const TensorImpl> a);
std::shared_ptr<TensorImpl> relu(std::shared_ptr<const TensorImpl> a);
std::shared_ptr<TensorImpl> softmax(std::shared_ptr<const TensorImpl> a, size_t dim = SIZE_MAX);
}
