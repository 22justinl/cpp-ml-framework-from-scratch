#pragma once

#include "core/tensor.h"

Tensor transpose(const Tensor& t1, size_t dim0, size_t dim1);
Tensor squeeze(const Tensor& t1, size_t dim);
Tensor unsqueeze(const Tensor& t1, size_t dim);
Tensor reshape(const Tensor& t1, const std::vector<size_t>& new_shape);
