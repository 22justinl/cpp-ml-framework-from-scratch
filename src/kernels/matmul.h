#pragma once
#include "core/tensor.h"

using std::shared_ptr;
namespace kernels {
shared_ptr<TensorImpl> matmul(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b);
}
