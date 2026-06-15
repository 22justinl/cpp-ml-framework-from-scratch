#include "tensor_ops.h"

#include "kernels/math_ops.h"
#include "kernels/tensor_ops.h"

TransposeOp::TransposeOp(const Tensor& t1, size_t dim0, size_t dim1, const Tensor& t2): dim0(dim0), dim1(dim1) {
    a = t1.impl();
    out = t2.impl();
}
Tensor TransposeOp::forward(const Tensor& t1, size_t dim0, size_t dim1) {
    return Tensor(kernels::transpose(t1.impl(), dim0, dim1));
}
void TransposeOp::backward() {
    if (a->requires_grad) {
        kernels::add_inplace(a->grad->impl(), kernels::transpose(out->grad->impl(), dim0, dim1));
    }
}
std::vector<std::shared_ptr<const TensorImpl>> TransposeOp::inputs() {
    return {a};
}

SqueezeOp::SqueezeOp(const Tensor& t1, size_t dim, const Tensor& t2): dim(dim) {
    a = t1.impl();
    out = t2.impl();
}
Tensor SqueezeOp::forward(const Tensor& t1, size_t dim) {
    return kernels::squeeze(t1.impl(), dim);
}
void SqueezeOp::backward() {
    if (a->requires_grad) {
        kernels::add_inplace(a->grad->impl(), kernels::unsqueeze(out->grad->impl(), dim));
    }
}
std::vector<std::shared_ptr<const TensorImpl>> SqueezeOp::inputs() {
    return {a};
}

UnsqueezeOp::UnsqueezeOp(const Tensor& t1, size_t dim, const Tensor& t2): dim(dim) {
    a = t1.impl();
    out = t2.impl();
}
Tensor UnsqueezeOp::forward(const Tensor& t1, size_t dim) {
    return kernels::unsqueeze(t1.impl(), dim);
}
void UnsqueezeOp::backward() {
    if (a->requires_grad) {
        kernels::add_inplace(a->grad->impl(), kernels::squeeze(out->grad->impl(), dim));
    }
}
std::vector<std::shared_ptr<const TensorImpl>> UnsqueezeOp::inputs() {
    return {a};
}

ReshapeOp::ReshapeOp(const Tensor& t1, const Tensor& t2) {
    a = t1.impl();
    out = t2.impl();
}
Tensor ReshapeOp::forward(const Tensor& t1, const std::vector<size_t>& new_shape) {
    return kernels::reshape(t1.impl(), new_shape);
}
void ReshapeOp::backward() {
    if (a->requires_grad) {
        kernels::add_inplace(a->grad->impl(), kernels::reshape(out->grad->impl(), a->shape));
    }
}
std::vector<std::shared_ptr<const TensorImpl>> ReshapeOp::inputs() {
    return {a};
}

SliceOp::SliceOp(const Tensor& t1, const std::vector<TensorIndex>& indices, const Tensor& t2): indices(indices) {
    a = t1.impl();
    out = t2.impl();
}
Tensor SliceOp::forward(const Tensor& t1, const std::vector<TensorIndex>& indices) {
    return kernels::slice(t1.impl(), indices);
}

void SliceOp::backward() {
    if (a->requires_grad) {
        kernels::add_inplace(kernels::slice(a->grad->impl(), indices), out->grad->impl());
    }
}
std::vector<std::shared_ptr<const TensorImpl>> SliceOp::inputs() {
    return {a};
}
