#include "reduction_ops.h"

#include "kernels/reduction_ops.h"
#include "kernels/math_ops.h"
#include "utils/tensor_utils.h"

// WARN: backward functions need broadcasting, current implementations are not ideal

SumOp::SumOp(const Tensor& t1, size_t dim, const Tensor& t2): dim(dim) {
    a = t1.impl();
    out = t2.impl();
}

Tensor SumOp::forward(const Tensor& t1, size_t dim) {
    return Tensor(kernels::sum(t1.impl(), dim));
}

void SumOp::backward() {
    if (a->requires_grad) {
        if (a->shape.size() == 0) {
            return;
        }
        if (a->shape.size() == 1 || dim == SIZE_T_MAX) {
            kernels::add_inplace(a->grad->impl(), kernels::scalar_mul(out->grad->data_raw()[0], create_tensorimpl(std::vector<float>(a->data.size(), 1), a->shape, a->strides, false)));
        } else if (a->shape.size() == 2) {
            throw std::runtime_error("Sum along dimension operation gradient not implemented");
        } else {
            throw std::runtime_error("Sum operation gradient for nD not implemented");
        }
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
    if (a->requires_grad) {
        if (a->shape.size() == 0) {
            return;
        }
        if (a->shape.size() == 1 || dim == SIZE_T_MAX) {
            size_t n = dim == SIZE_T_MAX ? a->data.size() : a->shape[dim];
            kernels::add_inplace(a->grad->impl(), kernels::scalar_mul(out->grad->data_raw()[0], create_tensorimpl(std::vector<float>(a->data.size(), 1.0/n), a->shape, a->strides, false)));
        } else if (a->shape.size() == 2) {
            throw std::runtime_error("Mean along dimension operation gradient not implemented");
        } else {
            throw std::runtime_error("Mean operation gradient for nD not implemented");
        }
    }
}
std::vector<std::shared_ptr<const TensorImpl>> MeanOp::inputs() {
    return {a};
}

MaxOp::MaxOp(const Tensor& t1, size_t dim, std::shared_ptr<std::vector<size_t>> indices_ptr, const Tensor& t2): dim(dim), indices_ptr(indices_ptr) {
    a = t1.impl();
    out = t2.impl();
}

Tensor MaxOp::forward(const Tensor& t1, size_t dim, std::shared_ptr<std::vector<size_t>>* indices_pptr) {
    return Tensor(kernels::max(t1.impl(), dim, indices_pptr));
}

void MaxOp::backward() {
    if (a->requires_grad) {
        if (a->shape.size() == 0) {
            return;
        }
        if (a->shape.size() == 1 || dim == SIZE_T_MAX) {
            std::vector<float> temp_data(a->data.size(), 0);
            temp_data[(*indices_ptr)[0]] = 1;
            std::shared_ptr<TensorImpl> temp = create_tensorimpl(temp_data, a->shape, a->strides, false);
            kernels::add_inplace(a->grad->impl(), kernels::scalar_mul(out->grad->data_raw()[0], temp));
        } else if (a->shape.size() == 2) {
            std::vector<float> temp_data(a->shape[1-dim], 0);
            std::vector<size_t> curr_idx = {0, 0};
            for (size_t i = 0; i < a->shape[1-dim]; ++i) {
                curr_idx[1-dim] = i;
                curr_idx[dim] = (*indices_ptr)[i];
                temp_data[calculate_offset(a, curr_idx)] = 1;
            }
            std::shared_ptr<TensorImpl> temp = create_tensorimpl(temp_data, a->shape, a->strides, false);
            kernels::add_inplace(a->grad->impl(), kernels::mul(temp, out->grad->impl()));
        } else {
            throw std::runtime_error("Max operation gradient for nD not implemented");
        }
    }
}
std::vector<std::shared_ptr<const TensorImpl>> MaxOp::inputs() {
    return {a};
}

MinOp::MinOp(const Tensor& t1, size_t dim, std::shared_ptr<std::vector<size_t>> indices_ptr, const Tensor& t2): dim(dim), indices_ptr(indices_ptr) {
    a = t1.impl();
    out = t2.impl();
}

Tensor MinOp::forward(const Tensor& t1, size_t dim, std::shared_ptr<std::vector<size_t>>* indices_pptr) {
    return Tensor(kernels::min(t1.impl(), dim, indices_pptr));
}

void MinOp::backward() {
    if (a->requires_grad) {
        if (a->shape.size() == 0) {
            return;
        }
        if (a->shape.size() == 1 || dim == SIZE_T_MAX) {
            std::vector<float> temp_data(a->data.size(), 0);
            temp_data[(*indices_ptr)[0]] = 1;
            std::shared_ptr<TensorImpl> temp = create_tensorimpl(temp_data, a->shape, a->strides, false);
            kernels::add_inplace(a->grad->impl(), kernels::scalar_mul(out->grad->data_raw()[0], temp));
        } else if (a->shape.size() == 2) {
            std::vector<float> temp_data(a->shape[1-dim], 0);
            std::vector<size_t> curr_idx = {0, 0};
            for (size_t i = 0; i < a->shape[1-dim]; ++i) {
                curr_idx[1-dim] = i;
                curr_idx[dim] = (*indices_ptr)[i];
                temp_data[calculate_offset(a, curr_idx)] = 1;
            }
            std::shared_ptr<TensorImpl> temp = create_tensorimpl(temp_data, a->shape, a->strides, false);
            kernels::add_inplace(a->grad->impl(), kernels::mul(temp, out->grad->impl()));
        } else {
            throw std::runtime_error("Min operation gradient for nD not implemented");
        }
    }
}
std::vector<std::shared_ptr<const TensorImpl>> MinOp::inputs() {
    return {a};
}
