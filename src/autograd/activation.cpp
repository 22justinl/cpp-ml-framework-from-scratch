#include "autograd/activation.h"

#include "kernels/activation.h"
#include "kernels/math_ops.h"
#include "utils/tensor_utils.h"

SigmoidOp::SigmoidOp(const Tensor& t1, const Tensor& t2) {
    a = t1.impl();
    out = t2.impl();
}
Tensor SigmoidOp::forward(const Tensor& t1) {
    return Tensor(kernels::sigmoid(t1.impl()));
}
void SigmoidOp::backward() {
    if (a->requires_grad) {
        kernels::add_inplace(a->grad->impl(), kernels::mul(out->grad->impl(), kernels::mul(out, kernels::sub(create_tensorimpl(std::vector<float>(out->data.size(), 1), out->shape, out->strides, false), out))));
        if (a->grad_fn) {
            a->grad_fn->backward();
        }
    }
}

ReLUOp::ReLUOp(const Tensor& t1, const Tensor& t2) {
    a = t1.impl();
    out = t2.impl();
}
Tensor ReLUOp::forward(const Tensor& t1) {
    return Tensor(kernels::relu(t1.impl()));
}
void ReLUOp::backward() {
    if (a->requires_grad) {
        std::vector<float> pos(a->data.size(), 0);
        for (size_t i = 0; i < a->data.size(); ++i) {
            if (a->data[i] > 0) {
                pos[i] = 1;
            }
        }
        kernels::add_inplace(a->grad->impl(), kernels::mul(create_tensorimpl(pos, a->shape, a->strides, false), out->grad->impl()));
        if (a->grad_fn) {
            a->grad_fn->backward();
        }
    }
}

SoftmaxOp::SoftmaxOp(const Tensor& t1, const Tensor& t2) {
    a = t1.impl();
    out = t2.impl();
}
Tensor SoftmaxOp::forward(const Tensor& t1) {
    return Tensor(kernels::softmax(t1.impl()));
}
void SoftmaxOp::backward() {
    if (a->requires_grad) {
        float d = kernels::dot(out->grad->impl(), out)->data[0];
        kernels::add_inplace(a->grad->impl(), kernels::mul(out, kernels::sub(out->grad->impl(), create_tensorimpl(std::vector(out->data.size(), d), out->shape, out->strides, false))));
        if (a->grad_fn) {
            a->grad_fn->backward();
        }
    }
}
