#include "reduction_ops.h"

#include "core/broadcast.h"
#include "kernels/reduction_ops.h"
#include "kernels/math_ops.h"
#include "utils/tensor_utils.h"

SumOp::SumOp(const Tensor& t1, size_t dim, const Tensor& t2): dim(dim) {
    a = t1.impl();
    out = t2.impl();
}
Tensor SumOp::forward(const Tensor& t1, size_t dim) {
    return Tensor(kernels::sum(t1.impl(), dim));
}
void SumOp::backward() {
    if (!a->requires_grad || a->shape.size() == 0) {
        return;
    }
    if (a->shape.size() == 1 || dim == SIZE_T_MAX) {
        kernels::add_inplace(a->grad->impl(), kernels::scalar_mul(out->grad->data_raw()[0], create_tensorimpl(std::vector<float>(a->data.size(), 1), a->shape, a->strides, false)));
    } else if (a->shape.size() == 2) {
        std::shared_ptr<TensorImpl> grad = create_tensorimpl(std::vector<float>(a->data.size(), 1), a->shape, a->strides, false);
        std::shared_ptr<TensorImpl> out_grad = dim == 1 ? vec_to_col(out->grad->impl()) : out->grad->impl();
        BroadcastInfo b_info = construct_broadcast_info(grad, out_grad);
        grad = kernels::mul_broadcast(grad, out_grad, b_info);
        kernels::add_inplace(a->grad->impl(), grad);
    } else {
        throw std::runtime_error("Sum operation gradient for nD not implemented");
    }
}
std::vector<std::shared_ptr<const TensorImpl>> SumOp::inputs() {
    return {a};
}

MeanOp::MeanOp(const Tensor& t1, size_t dim, const Tensor& t2): dim(dim) {
    a = t1.impl();
    out = t2.impl();
}
Tensor MeanOp::forward(const Tensor& t1, size_t dim) {
    return Tensor(kernels::mean(t1.impl(), dim));
}
void MeanOp::backward() {
    if (!a->requires_grad || a->shape.size() == 0) {
        return;
    }
    if (a->shape.size() == 1 || dim == SIZE_T_MAX) {
        size_t n = dim == SIZE_T_MAX ? a->data.size() : a->shape[dim];
        kernels::add_inplace(a->grad->impl(), kernels::scalar_mul(out->grad->data_raw()[0], create_tensorimpl(std::vector<float>(a->data.size(), 1.0/n), a->shape, a->strides, false)));
    } else if (a->shape.size() == 2) {
        size_t n = a->shape[dim];
        std::shared_ptr<TensorImpl> grad = create_tensorimpl(std::vector<float>(a->data.size(), 1.0/n), a->shape, a->strides, false);
        std::shared_ptr<TensorImpl> out_grad = dim == 1 ? vec_to_col(out->grad->impl()) : out->grad->impl();
        BroadcastInfo b_info = construct_broadcast_info(grad, out_grad);
        grad = kernels::mul_broadcast(grad, out_grad, b_info);
        kernels::add_inplace(a->grad->impl(), grad);
    } else {
        throw std::runtime_error("Mean operation gradient for nD not implemented");
    }
}
std::vector<std::shared_ptr<const TensorImpl>> MeanOp::inputs() {
    return {a};
}

MaxOp::MaxOp(const Tensor& t1, size_t dim, std::shared_ptr<std::vector<size_t>> offsets_ptr, const Tensor& t2): dim(dim), offsets_ptr(offsets_ptr) {
    a = t1.impl();
    out = t2.impl();
}
Tensor MaxOp::forward(const Tensor& t1, size_t dim, std::shared_ptr<std::vector<size_t>>* offsets_pptr) {
    return Tensor(kernels::max(t1.impl(), dim, false, offsets_pptr));
}
void MaxOp::backward() {
    if (!a->requires_grad || a->shape.size() == 0) {
        return;
    }
    if (a->shape.size() == 1 || dim == SIZE_T_MAX) {
        std::shared_ptr<TensorImpl> temp = create_tensorimpl(std::vector<float>(a->data.size(), 0), a->shape, a->strides, false);
        temp->data[(*offsets_ptr)[0]] = 1;
        kernels::add_inplace(a->grad->impl(), kernels::scalar_mul(out->grad->data_raw()[0], temp));
    } else if (a->shape.size() == 2) {
        std::shared_ptr<TensorImpl> temp = create_tensorimpl(std::vector<float>(a->data.size(), 0), a->shape, a->strides, false);
        std::shared_ptr<TensorImpl> out_grad = dim == 1 ? vec_to_col(out->grad->impl()) : out->grad->impl();
        for (size_t i = 0; i < a->shape[1-dim]; ++i) {
            temp->data[(*offsets_ptr)[i]] = 1;
        }
        BroadcastInfo b_info = construct_broadcast_info(temp, out_grad);
        temp = kernels::mul_broadcast(temp, out_grad, b_info);
        kernels::add_inplace(a->grad->impl(), temp);
    } else {
        throw std::runtime_error("Max operation gradient for nD not implemented");
    }
}
std::vector<std::shared_ptr<const TensorImpl>> MaxOp::inputs() {
    return {a};
}

MinOp::MinOp(const Tensor& t1, size_t dim, std::shared_ptr<std::vector<size_t>> offsets_ptr, const Tensor& t2): dim(dim), offsets_ptr(offsets_ptr) {
    a = t1.impl();
    out = t2.impl();
}
Tensor MinOp::forward(const Tensor& t1, size_t dim, std::shared_ptr<std::vector<size_t>>* offsets_pptr) {
    return Tensor(kernels::min(t1.impl(), dim, false, offsets_pptr));
}
void MinOp::backward() {
    if (!a->requires_grad || a->shape.size() == 0) {
        return;
    }
    if (a->shape.size() == 1 || dim == SIZE_T_MAX) {
        std::vector<float> temp_data(a->data.size(), 0);
        temp_data[(*offsets_ptr)[0]] = 1;
        std::shared_ptr<TensorImpl> temp = create_tensorimpl(temp_data, a->shape, a->strides, false);
        kernels::add_inplace(a->grad->impl(), kernels::scalar_mul(out->grad->data_raw()[0], temp));
    } else if (a->shape.size() == 2) {
        std::shared_ptr<TensorImpl> temp = create_tensorimpl(std::vector<float>(a->data.size(), 0), a->shape, a->strides, false);
        std::shared_ptr<TensorImpl> out_grad = dim == 1 ? vec_to_col(out->grad->impl()) : out->grad->impl();
        for (size_t i = 0; i < a->shape[1-dim]; ++i) {
            temp->data[(*offsets_ptr)[i]] = 1;
        }
        BroadcastInfo b_info = construct_broadcast_info(temp, out_grad);
        temp = kernels::mul_broadcast(temp, out_grad, b_info);
        kernels::add_inplace(a->grad->impl(), temp);
    } else {
        throw std::runtime_error("Min operation gradient for nD not implemented");
    }
}
std::vector<std::shared_ptr<const TensorImpl>> MinOp::inputs() {
    return {a};
}
