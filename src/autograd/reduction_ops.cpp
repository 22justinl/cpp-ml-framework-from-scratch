#include "reduction_ops.h"

#include "kernels/reduction_ops.h"
#include "kernels/math_ops.h"

SumOp::SumOp(const Tensor& t1, size_t dim, const Tensor& t2) {
    a = t1.impl();
    out = t2.impl();
}

Tensor SumOp::forward(const Tensor& t1, size_t dim) {
    return Tensor(kernels::sum(t1.impl(), dim));
}

void SumOp::backward() {
    if (a->requires_grad) {
        kernels::add_inplace(a->grad->impl(), out->grad->impl());
        if (a->grad_fn) {
            a->grad_fn->backward();
        }
    }
}

MeanOp::MeanOp(const Tensor& t1, size_t dim, const Tensor& t2): dim(dim) {
    a = t1.impl();
    out = t2.impl();
}

Tensor MeanOp::forward(const Tensor& t1, size_t dim) {
    return Tensor(kernels::mean(t1.impl(), dim));
}

void MeanOp::backward() {
    if (a->requires_grad) {
        size_t n = dim == SIZE_T_MAX ? a->data.size() : a->shape[dim];
        kernels::add_inplace(a->grad->impl(), kernels::scalar_mul(1.0/n, out->grad->impl()));
        if (a->grad_fn) {
            a->grad_fn->backward();
        }
    }
}

// MaxOp::MaxOp(const Tensor& t1, size_t dim, const Tensor& t2) {
//
// }
//
// Tensor MaxOp::forward(const Tensor& t1, size_t dim) {
//
// }
//
// void MaxOp::backward() {
//
// }
//
//
// MinOp::MinOp(const Tensor& t1, size_t dim, const Tensor& t2) {
//
// }
//
// Tensor MinOp::forward(const Tensor& t1, size_t dim) {
//
// }
//
// void MinOp::backward() {
//
// }
