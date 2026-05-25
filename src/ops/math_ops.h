#pragma once

#include "core/tensor.h"

// Element-wise addition
Tensor add(const Tensor& a, const Tensor& b);
// Element-wise subtraction
Tensor sub(const Tensor& a, const Tensor& b);
// Element-wise multiplication
Tensor mul(const Tensor& a, const Tensor& b);
// Element-wise division
Tensor div(const Tensor& a, const Tensor& b);

// Tensor multiplication: (m, n)(n, l)
Tensor matmul(const Tensor& a, const Tensor& b);

// Tensor multiplication (m, n)(n) or (m, n)(n, 1)
Tensor matvec(const Tensor& a, const Tensor& b);

// (n)(n) or (n,1)(n,1) or (1,n)(1,n)
Tensor dot(const Tensor& a, const Tensor& b);

Tensor transpose(const Tensor& t);

// TODO: exp, log, power, sqrt, sin, cos, tan
