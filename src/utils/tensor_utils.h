#pragma once

#include "core/tensor.h"

bool check_tensor_equal(const Tensor& t1, const Tensor& t2);
bool check_tensor_shape_match(const Tensor& t1, const Tensor& t2);
void print_tensor(const Tensor& t1);
