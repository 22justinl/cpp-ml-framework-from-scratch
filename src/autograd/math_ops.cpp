#include "autograd/math_ops.h"
#include "kernels/math_ops.h"
#include "utils/tensor_utils.h"
#include <iostream>

AddOp::AddOp(const Tensor& t1, const Tensor& t2, const Tensor& t3) {
    a = t1.impl();
    b = t2.impl();
    out = t3.impl();
}
Tensor AddOp::forward(const Tensor& t1, const Tensor& t2) {
    return Tensor(kernels::add(t1.impl(), t2.impl()));
}
void AddOp::backward() {
    if (a->requires_grad) {
        kernels::add_inplace(a->grad->impl(), out->grad->impl());
        if (a->grad_fn) {
            a->grad_fn->backward();
        }
    }
    if (b->requires_grad) {
        kernels::add_inplace(b->grad->impl(), out->grad->impl());
        if (b->grad_fn) {
            b->grad_fn->backward();
        }
    }
}

SubOp::SubOp(const Tensor& t1, const Tensor& t2, const Tensor& t3) {
    a = t1.impl();
    b = t2.impl();
    out = t3.impl();
}
Tensor SubOp::forward(const Tensor& t1, const Tensor& t2) {
    return Tensor(kernels::sub(t1.impl(), t2.impl()));
}
void SubOp::backward() {
    if (a->requires_grad) {
        kernels::add_inplace(a->grad->impl(), out->grad->impl());
        if (a->grad_fn) {
            a->grad_fn->backward();
        }
    }
    if (b->requires_grad) {
        kernels::sub_inplace(b->grad->impl(), out->grad->impl());
        if (b->grad_fn) {
            b->grad_fn->backward();
        }
    }
}

MulOp::MulOp(const Tensor& t1, const Tensor& t2, const Tensor& t3) {
    a = t1.impl();
    b = t2.impl();
    out = t3.impl();
}
Tensor MulOp::forward(const Tensor& t1, const Tensor& t2) {
    return Tensor(kernels::mul(t1.impl(), t2.impl()));
}
void MulOp::backward() {
    if (a->requires_grad) {
        kernels::add_inplace(a->grad->impl(), kernels::mul(b, out->grad->impl()));
        if (a->grad_fn) {
            a->grad_fn->backward();
        }
    }
    if (b->requires_grad) {
        kernels::add_inplace(b->grad->impl(), kernels::mul(a, out->grad->impl()));
        if (b->grad_fn) {
            b->grad_fn->backward();
        }
    }
}

DivOp::DivOp(const Tensor& t1, const Tensor& t2, const Tensor& t3) {
    a = t1.impl();
    b = t2.impl();
    out = t3.impl();
}
Tensor DivOp::forward(const Tensor& t1, const Tensor& t2) {
    return Tensor(kernels::div(t1.impl(), t2.impl()));
}
void DivOp::backward() {
    std::shared_ptr<TensorImpl> divb = kernels::div(out->grad->impl(), b);
    if (a->requires_grad) {
        kernels::add_inplace(a->grad->impl(), divb);
        if (a->grad_fn) {
            a->grad_fn->backward();
        }
    }
    if (b->requires_grad) {
        kernels::sub_inplace(b->grad->impl(), kernels::div(kernels::mul(divb, a), b));
        if (b->grad_fn) {
            b->grad_fn->backward();
        }
    }
}

ScalarMulOp::ScalarMulOp(float f, const Tensor& t1, const Tensor& t2): f(f) {
    a = t1.impl();
    out = t2.impl();
}
Tensor ScalarMulOp::forward(float f, const Tensor& t1) {
    return Tensor(kernels::scalar_mul(f, t1.impl()));
}
void ScalarMulOp::backward() {
    if (a->requires_grad) {
        kernels::add_inplace(a->grad->impl(), kernels::scalar_mul(f, out->grad->impl()));
        if (a->grad_fn) {
            a->grad_fn->backward();
        }
    }
}

MatMulOp::MatMulOp(const Tensor& t1, const Tensor& t2, const Tensor& t3) {
    a = t1.impl();
    b = t2.impl();
    out = t3.impl();
}
Tensor MatMulOp::forward(const Tensor& t1, const Tensor& t2) {
    return Tensor(kernels::matmul(t1.impl(), t2.impl()));
}
void MatMulOp::backward() {
    if (a->requires_grad) {
        kernels::add_inplace(a->grad->impl(), kernels::matmul(out->grad->impl(), kernels::transpose(b)));
        if (a->grad_fn) {
            a->grad_fn->backward();
        }
    }
    if (b->requires_grad) {
        kernels::add_inplace(b->grad->impl(), kernels::matmul(kernels::transpose(a), out->grad->impl()));
        if (b->grad_fn) {
            b->grad_fn->backward();
        }
    }
}

MatVecOp::MatVecOp(const Tensor& t1, const Tensor& t2, const Tensor& t3) {
    a = t1.impl();
    b = t2.impl();
    out = t3.impl();
}
Tensor MatVecOp::forward(const Tensor& t1, const Tensor& t2) {
    return Tensor(kernels::matvec(t1.impl(), t2.impl()));
}
void MatVecOp::backward() {
    if (a->requires_grad) {
        kernels::add_inplace(a->grad->impl(), kernels::matmul(out->grad->impl(), kernels::transpose(b)));
        if (a->grad_fn) {
            a->grad_fn->backward();
        }
    }
    if (b->requires_grad) {
        kernels::add_inplace(b->grad->impl(), col_to_1d(kernels::matmul(kernels::transpose(a), out->grad->impl())));
        if (b->grad_fn) {
            b->grad_fn->backward();
        }
    }
}

DotOp::DotOp(const Tensor& t1, const Tensor& t2, const Tensor& t3) {
    a = t1.impl();
    b = t2.impl();
    out = t3.impl();
}
Tensor DotOp::forward(const Tensor& t1, const Tensor& t2) {
    return Tensor(kernels::dot(t1.impl(), t2.impl()));
}
void DotOp::backward() {
    if (a->requires_grad) {
        kernels::add_inplace(a->grad->impl(), kernels::scalar_mul(out->grad->impl()->data[0], b));
        if (a->grad_fn) {
            a->grad_fn->backward();
        }
    }
    if (b->requires_grad) {
        kernels::add_inplace(b->grad->impl(), kernels::scalar_mul(out->grad->impl()->data[0], a));
        if (b->grad_fn) {
            b->grad_fn->backward();
        }
    }
}

TransposeOp::TransposeOp(const Tensor& t1, const Tensor& t2) {
    a = t1.impl();
    out = t2.impl();
}
Tensor TransposeOp::forward(const Tensor& t1) {
    return Tensor(kernels::transpose(t1.impl()));
}
void TransposeOp::backward() {
    if (a->requires_grad) {
        kernels::add_inplace(a->grad->impl(), kernels::transpose(out->grad->impl()));
    }
}
