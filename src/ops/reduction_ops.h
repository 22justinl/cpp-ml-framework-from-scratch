#pragma once

#include "core/tensor.h"

Tensor sum(const Tensor& t1, size_t dim=SIZE_MAX, bool keepdim=false);
Tensor mean(const Tensor& t1, size_t dim=SIZE_MAX, bool keepdim=false);
Tensor max(const Tensor& t1, size_t dim=SIZE_MAX, bool keepdim=false);
Tensor min(const Tensor& t1, size_t dim=SIZE_MAX, bool keepdim=false);
