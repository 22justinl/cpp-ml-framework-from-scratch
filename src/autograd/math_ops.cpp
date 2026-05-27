#include "autograd/math_ops.h"
#include "utils/tensor_utils.h"

AddOp::AddOp(const Tensor& t1, const Tensor& t2, const Tensor& t3) {
    a = t1.impl();
    b = t2.impl();
    out = t3.impl();
}

Tensor AddOp::forward(const Tensor& a, const Tensor& b) {
    if (!check_tensor_shape_match(a, b)) {
        throw std::runtime_error("Tensor dimension mismatch: " + a.shape_string() + " and " + b.shape_string());
    }
    Tensor res = Tensor(0, a.shape(), a.requires_grad() || b.requires_grad());
    const std::vector<float>& a_data = a.data_raw();
    const std::vector<float>& b_data = b.data_raw();
    std::vector<float>& res_data = res.data_raw();

    for (size_t i = 0; i < a_data.size(); ++i) {
        res_data[i] = a_data[i] + b_data[i];
    }
    return res;
}

void AddOp::backward() {
    if (a->requires_grad) {
        *a->grad += *out->grad;
        if (a->grad_fn) {
            a->grad_fn->backward();
        }
    }
    if (b->requires_grad) {
        *b->grad += *out->grad;
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

Tensor MulOp::forward(const Tensor& a, const Tensor& b) {
    if (!check_tensor_shape_match(a, b)) {
        throw std::runtime_error("Tensor dimension mismatch: " + a.shape_string() + " and " + b.shape_string());
    }
    Tensor res = Tensor(0, a.shape(), a.requires_grad() || b.requires_grad());
    const std::vector<float>& a_data = a.data_raw();
    const std::vector<float>& b_data = b.data_raw();
    std::vector<float>& res_data = res.data_raw();

    for (size_t i = 0; i < a_data.size(); ++i) {
        res_data[i] = a_data[i] * b_data[i];
    }
    return res;
}

void MulOp::backward() {
    if (a->requires_grad) {
        *a->grad += *out->grad * Tensor(b->data, b->shape);
        if (a->grad_fn) {
            a->grad_fn->backward();
        }
    }
    if (b->requires_grad) {
        *b->grad += *out->grad * Tensor(a->data, a->shape);
        if (b->grad_fn) {
            b->grad_fn->backward();
        }
    }
}
