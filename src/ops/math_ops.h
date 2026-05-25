#pragma once

#include "core/tensor.h"

// Tensor multiplication: (m, n)(n, l)
Tensor matmul(const Tensor& a, const Tensor& b);

// Tensor multiplication (m, n)(n) or (m, n)(n, 1)
Tensor matvec(const Tensor& a, const Tensor& b);

// (n)(n) or (n,1)(n,1) or (1,n)(1,n)
Tensor dot(const Tensor& a, const Tensor& b);

Tensor transpose(const Tensor& t);

// TODO: exp, log, power, sqrt, sin, cos, tan
