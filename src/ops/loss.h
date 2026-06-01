#pragma once

#include "core/tensor.h"

Tensor mse_loss(const Tensor& pred, const Tensor& target);
Tensor cross_entropy_loss(const Tensor& pred, const Tensor& target);
