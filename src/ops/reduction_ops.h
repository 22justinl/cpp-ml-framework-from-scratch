#pragma once

#include "core/tensor.h"

Tensor sum(const Tensor& t1, size_t dim=SIZE_T_MAX);
Tensor mean(const Tensor& t1, size_t dim=SIZE_T_MAX);
Tensor max(const Tensor& t1, size_t dim=SIZE_T_MAX);
Tensor min(const Tensor& t1, size_t dim=SIZE_T_MAX);
