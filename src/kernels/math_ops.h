#pragma once
#include "core/tensor.h"

namespace kernels {
void add_inplace(std::shared_ptr<TensorImpl> a, std::shared_ptr<const TensorImpl> b);
void sub_inplace(std::shared_ptr<TensorImpl> a, std::shared_ptr<const TensorImpl> b);
std::shared_ptr<TensorImpl> add(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b);
std::shared_ptr<TensorImpl> sub(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b);
std::shared_ptr<TensorImpl> mul(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b);
std::shared_ptr<TensorImpl> div(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b);
std::shared_ptr<TensorImpl> matmul(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b);
std::shared_ptr<TensorImpl> transpose(std::shared_ptr<const TensorImpl> a);
}
