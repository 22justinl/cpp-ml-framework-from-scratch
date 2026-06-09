#pragma once

#include "core/tensor.h"

namespace kernels {
std::shared_ptr<TensorImpl> sum(std::shared_ptr<const TensorImpl> a, size_t dim = SIZE_T_MAX, bool keepdim=false);
std::shared_ptr<TensorImpl> mean(std::shared_ptr<const TensorImpl> a, size_t dim = SIZE_T_MAX, bool keepdim=false);
std::shared_ptr<TensorImpl> max(std::shared_ptr<const TensorImpl> a, size_t dim = SIZE_T_MAX, bool keepdim=false, std::shared_ptr<TensorImpl>* max_pos_ptr = nullptr);
std::shared_ptr<TensorImpl> min(std::shared_ptr<const TensorImpl> a, size_t dim = SIZE_T_MAX, bool keepdim=false, std::shared_ptr<TensorImpl>* min_pos_ptr = nullptr);
}
