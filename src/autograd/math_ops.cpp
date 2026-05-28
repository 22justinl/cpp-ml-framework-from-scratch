#include "autograd/math_ops.h"
#include "utils/tensor_utils.h"
#include "kernels/math_ops.h"

AddOp::AddOp(const Tensor& t1, const Tensor& t2, const Tensor& t3) {
    a = t1.impl();
    b = t2.impl();
    out = t3.impl();
}
Tensor AddOp::forward(const Tensor& t1, const Tensor& t2) {
    if (!check_tensor_shape_match(t1, t2)) {
        throw std::runtime_error("Tensor dimension mismatch: " + t1.shape_string() + " and " + t2.shape_string());
    }
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
    if (!check_tensor_shape_match(t1, t2)) {
        throw std::runtime_error("Tensor dimension mismatch: " + t1.shape_string() + " and " + t2.shape_string());
    }
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
    if (!check_tensor_shape_match(t1, t2)) {
        throw std::runtime_error("Tensor dimension mismatch: " + t1.shape_string() + " and " + t2.shape_string());
    }
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
    if (!check_tensor_shape_match(t1, t2)) {
        throw std::runtime_error("Tensor dimension mismatch: " + t1.shape_string() + " and " + t2.shape_string());
    }
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

MatmulOp::MatmulOp(const Tensor& t1, const Tensor& t2, const Tensor& t3) {
    a = t1.impl();
    b = t2.impl();
    out = t3.impl();
}
Tensor MatmulOp::forward(const Tensor& t1, const Tensor& t2) {
    // TODO: nD matmul
    if (t1.shape()[t1.shape().size()-1] != t2.shape()[0]) {
        throw std::runtime_error("Cannot matrix multiply tensors with shapes " + t1.shape_string() + " and " + t2.shape_string());
    }
    return Tensor(kernels::matmul(t1.impl(), t2.impl()));
}
void MatmulOp::backward() {
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
