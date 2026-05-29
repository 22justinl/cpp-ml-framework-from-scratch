#pragma once

#include "core/tensor.h"

// Element-wise addition
Tensor add(const Tensor& t1, const Tensor& t2);
// Element-wise subtraction
Tensor sub(const Tensor& t1, const Tensor& t2);
// Element-wise multiplication
Tensor mul(const Tensor& t1, const Tensor& t2);
// Element-wise division
Tensor div(const Tensor& t1, const Tensor& t2);
// Scalar multiplication
Tensor scalar_mul(const Tensor& t1, float f);
Tensor scalar_mul(float f, const Tensor& t1);

// Tensor multiplication: (m, n)(n, l)
Tensor matmul(const Tensor& t1, const Tensor& t2);

// Tensor multiplication (m, n)(n) or (m, n)(n, 1)
Tensor matvec(const Tensor& t1, const Tensor& t2);

// (n)(n) or (n,1)(n,1) or (1,n)(1,n)
Tensor dot(const Tensor& t1, const Tensor& t2);

Tensor transpose(const Tensor& t);

// TODO: exp, log, power, sqrt, sin, cos, tan
