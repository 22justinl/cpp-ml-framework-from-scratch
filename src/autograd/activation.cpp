#include "autograd/activation.h"

#include "core/broadcast.h"
#include "kernels/activation.h"
#include "kernels/math_ops.h"
#include "kernels/reduction_ops.h"

using std::shared_ptr;
using std::make_shared;

SigmoidOp::SigmoidOp(const Tensor& t1, const Tensor& t2) {
    a = t1.impl();
    out = t2.impl();
}
Tensor SigmoidOp::forward(const Tensor& t1) {
    return Tensor(kernels::sigmoid(t1.impl()));
}
void SigmoidOp::backward() {
    if (a->requires_grad) {
        kernels::add_inplace(a->grad->impl(), kernels::mul(out->grad->impl(), kernels::mul(out, kernels::sub(make_shared<TensorImpl>(1, out->shape, out->strides, false), out))));
    }
}
std::vector<shared_ptr<const TensorImpl>> SigmoidOp::inputs() {
    return {a};
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
        std::vector<float> pos(a->storage->data.size(), 0);
        for (size_t i = 0; i < a->storage->data.size(); ++i) {
            if (a->storage->data[i] > 0) {
                pos[i] = 1;
            }
        }
        kernels::add_inplace(a->grad->impl(), kernels::mul(make_shared<TensorImpl>(pos, a->shape, a->strides, false), out->grad->impl()));
    }
}
std::vector<shared_ptr<const TensorImpl>> ReLUOp::inputs() {
    return {a};
}

SoftmaxOp::SoftmaxOp(const Tensor& t1, size_t dim, const Tensor& t2): dim(dim) {
    a = t1.impl();
    out = t2.impl();
}
Tensor SoftmaxOp::forward(const Tensor& t1, size_t dim = SIZE_MAX) {
    return Tensor(kernels::softmax(t1.impl(), dim));
}
void SoftmaxOp::backward() {
    if (a->requires_grad) {
        shared_ptr<TensorImpl> d = kernels::sum(kernels::mul(out->grad->impl(), out), dim, true);
        BroadcastInfo b_info = construct_broadcast_info(out->grad->impl(), d);
        kernels::add_inplace(a->grad->impl(), kernels::mul(out, kernels::sub_broadcast(out->grad->impl(), d, b_info)));
    }
}
std::vector<shared_ptr<const TensorImpl>> SoftmaxOp::inputs() {
    return {a};
}
