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
Tensor scaled_matmul(float alpha, const Tensor& t1, const Tensor& t2);
Tensor mmadd(const Tensor& t1, const Tensor& t2, const Tensor& t3);
Tensor mmadd_general(float alpha, const Tensor& t1, const Tensor& t2, float beta, const Tensor& t3);

// (n)(n) or (n,1)(n,1) or (1,n)(1,n)
Tensor dot(const Tensor& t1, const Tensor& t2);

Tensor power(const Tensor& t1, float f);
Tensor exp(const Tensor& t1);
Tensor log_e(const Tensor& t1);

// TODO: sin, cos, tan
