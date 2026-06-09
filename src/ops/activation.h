#pragma once

#include "core/tensor.h"

Tensor sigmoid(const Tensor& t1);
Tensor relu(const Tensor& t1);
Tensor softmax(const Tensor& t1, size_t dim=SIZE_MAX);
