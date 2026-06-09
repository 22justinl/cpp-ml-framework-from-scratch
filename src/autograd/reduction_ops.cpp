#include "reduction_ops.h"

#include "core/broadcast.h"
#include "kernels/reduction_ops.h"
#include "kernels/math_ops.h"

using std::shared_ptr;
using std::make_shared;

SumOp::SumOp(const Tensor& t1, size_t dim, bool keepdim, const Tensor& t2): dim(dim) {
    a = t1.impl();
    out = t2.impl();
}
Tensor SumOp::forward(const Tensor& t1, size_t dim, bool keepdim) {
    return Tensor(kernels::sum(t1.impl(), dim, keepdim));
}
void SumOp::backward() {
    if (!a->requires_grad || a->shape.size() == 0) {
        return;
    }
    if (a->shape.size() == 1 || dim == SIZE_MAX) {
        kernels::add_inplace(a->grad->impl(), kernels::scalar_mul(out->grad->data_raw()[0], make_shared<TensorImpl>(1, a->shape, a->strides, false)));
    } else {
        shared_ptr<TensorImpl> grad = make_shared<TensorImpl>(1, a->shape, a->strides, a->requires_grad);
        shared_ptr<TensorImpl> out_grad = dim == 1 ? vec_to_col(out->grad->impl()) : out->grad->impl();
        BroadcastInfo b_info = construct_broadcast_info(grad, out_grad);
        grad = kernels::mul_broadcast(grad, out_grad, b_info);
        kernels::add_inplace(a->grad->impl(), grad);
    }
}
std::vector<shared_ptr<const TensorImpl>> SumOp::inputs() {
    return {a};
}

MeanOp::MeanOp(const Tensor& t1, size_t dim, bool keepdim, const Tensor& t2): dim(dim) {
    a = t1.impl();
    out = t2.impl();
}
Tensor MeanOp::forward(const Tensor& t1, size_t dim, bool keepdim) {
    return Tensor(kernels::mean(t1.impl(), dim, keepdim));
}
void MeanOp::backward() {
    if (!a->requires_grad || a->shape.size() == 0) {
        return;
    }
    if (a->shape.size() == 1 || dim == SIZE_MAX) {
        size_t n = dim == SIZE_MAX ? a->storage->data.size() : a->shape[dim];
        kernels::add_inplace(a->grad->impl(), kernels::scalar_mul(out->grad->data_raw()[0], make_shared<TensorImpl>(1.0/n, a->shape, a->strides, false)));
    } else {
        size_t n = a->shape[dim];
        shared_ptr<TensorImpl> grad = make_shared<TensorImpl>(1.0/n, a->shape, a->strides, false);
        shared_ptr<TensorImpl> out_grad = dim == 1 ? vec_to_col(out->grad->impl()) : out->grad->impl();
        BroadcastInfo b_info = construct_broadcast_info(grad, out_grad);
        grad = kernels::mul_broadcast(grad, out_grad, b_info);
        kernels::add_inplace(a->grad->impl(), grad);
    }
}
std::vector<shared_ptr<const TensorImpl>> MeanOp::inputs() {
    return {a};
}

MaxOp::MaxOp(const Tensor& t1, size_t dim, bool keepdim, shared_ptr<TensorImpl> max_pos, const Tensor& t2): dim(dim), max_pos(max_pos) {
    a = t1.impl();
    out = t2.impl();
}
Tensor MaxOp::forward(const Tensor& t1, size_t dim, bool keepdim, shared_ptr<TensorImpl>* max_pos_ptr) {
    return Tensor(kernels::max(t1.impl(), dim, keepdim, max_pos_ptr));
}
void MaxOp::backward() {
    if (!a->requires_grad || a->shape.size() == 0) {
        return;
    }
    if (a->shape.size() == 1 || dim == SIZE_MAX) {
        kernels::add_inplace(a->grad->impl(), kernels::scalar_mul(out->grad->data_raw()[0], max_pos));
    } else {
        // if input is 2D with keepdim=false, we need to distinguish between dim=0 and dim=1 since output is 1D tensor
        shared_ptr<TensorImpl> out_grad = out->shape.size() == 1 && dim == 1 ? vec_to_col(out->grad->impl()) : out->grad->impl();
        BroadcastInfo b_info = construct_broadcast_info(max_pos, out_grad);
        kernels::add_inplace(a->grad->impl(), kernels::mul_broadcast(max_pos, out_grad, b_info));
    }
}
std::vector<shared_ptr<const TensorImpl>> MaxOp::inputs() {
    return {a};
}

MinOp::MinOp(const Tensor& t1, size_t dim, bool keepdim, shared_ptr<TensorImpl> min_pos, const Tensor& t2): dim(dim), min_pos(min_pos) {
    a = t1.impl();
    out = t2.impl();
}
Tensor MinOp::forward(const Tensor& t1, size_t dim, bool keepdim, shared_ptr<TensorImpl>* min_pos_ptr) {
    return Tensor(kernels::min(t1.impl(), dim, keepdim, min_pos_ptr));
}
void MinOp::backward() {
    if (!a->requires_grad || a->shape.size() == 0) {
        return;
    }
    if (a->shape.size() == 1 || dim == SIZE_MAX) {
        kernels::add_inplace(a->grad->impl(), kernels::scalar_mul(out->grad->data_raw()[0], min_pos));
    } else {
        // if input is 2D with keepdim=false, we need to distinguish between dim=0 and dim=1 since output is 1D tensor
        shared_ptr<TensorImpl> out_grad = out->shape.size() == 1 && dim == 1 ? vec_to_col(out->grad->impl()) : out->grad->impl();
        BroadcastInfo b_info = construct_broadcast_info(min_pos, out_grad);
        kernels::add_inplace(a->grad->impl(), kernels::mul_broadcast(min_pos, out_grad, b_info));
    }
}
std::vector<shared_ptr<const TensorImpl>> MinOp::inputs() {
    return {a};
}
