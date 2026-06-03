#include "autograd/math_ops.h"
#include "kernels/math_ops.h"
#include "kernels/reduction_ops.h"
#include "core/broadcast.h"
#include "utils/tensor_utils.h"

AddOp::AddOp(const Tensor& t1, const Tensor& t2, const Tensor& t3) {
    a = t1.impl();
    b = t2.impl();
    out = t3.impl();
}
Tensor AddOp::forward(const Tensor& t1, const Tensor& t2) {
    if (check_shape_match(t1.shape(), t2.shape())) {
        return Tensor(kernels::add(t1.impl(), t2.impl()));
    }
    BroadcastInfo b_info = construct_broadcast_info(t1.impl(), t2.impl());
    return Tensor(kernels::add_broadcast(t1.impl(), t2.impl(), b_info));
}
void AddOp::backward() {
    if (a->requires_grad) {
        kernels::add_inplace(a->grad->impl(), reduce_to_shape(out->grad->impl(), a->shape));
    }
    if (b->requires_grad) {
        kernels::add_inplace(b->grad->impl(), reduce_to_shape(out->grad->impl(), b->shape));
    }
}
std::vector<std::shared_ptr<const TensorImpl>> AddOp::inputs() {
    return {a, b};
}

SubOp::SubOp(const Tensor& t1, const Tensor& t2, const Tensor& t3) {
    a = t1.impl();
    b = t2.impl();
    out = t3.impl();
}
Tensor SubOp::forward(const Tensor& t1, const Tensor& t2) {
    if (check_shape_match(t1.shape(), t2.shape())) {
        return Tensor(kernels::sub(t1.impl(), t2.impl()));
    }
    BroadcastInfo b_info = construct_broadcast_info(t1.impl(), t2.impl());
    return Tensor(kernels::sub_broadcast(t1.impl(), t2.impl(), b_info));
}
void SubOp::backward() {
    if (a->requires_grad) {
        kernels::add_inplace(a->grad->impl(), reduce_to_shape(out->grad->impl(), a->shape));
    }
    if (b->requires_grad) {
        kernels::sub_inplace(b->grad->impl(), reduce_to_shape(out->grad->impl(), b->shape));
    }
}
std::vector<std::shared_ptr<const TensorImpl>> SubOp::inputs() {
    return {a, b};
}

MulOp::MulOp(const Tensor& t1, const Tensor& t2, const Tensor& t3) {
    a = t1.impl();
    b = t2.impl();
    out = t3.impl();
}
Tensor MulOp::forward(const Tensor& t1, const Tensor& t2) {
    if (check_shape_match(t1.shape(), t2.shape())) {
        return Tensor(kernels::mul(t1.impl(), t2.impl()));
    }
    BroadcastInfo b_info = construct_broadcast_info(t1.impl(), t2.impl());
    return Tensor(kernels::mul_broadcast(t1.impl(), t2.impl(), b_info));
}
void MulOp::backward() {
    if (a->requires_grad) {
        BroadcastInfo b_info = construct_broadcast_info(b, out->grad->impl());
        kernels::add_inplace(a->grad->impl(), reduce_to_shape(kernels::mul_broadcast(b, out->grad->impl(), b_info), a->shape));
    }
    if (b->requires_grad) {
        BroadcastInfo b_info = construct_broadcast_info(a, out->grad->impl());
        kernels::add_inplace(b->grad->impl(), reduce_to_shape(kernels::mul_broadcast(a, out->grad->impl(), b_info), b->shape));
    }
}
std::vector<std::shared_ptr<const TensorImpl>> MulOp::inputs() {
    return {a, b};
}

DivOp::DivOp(const Tensor& t1, const Tensor& t2, const Tensor& t3) {
    a = t1.impl();
    b = t2.impl();
    out = t3.impl();
}
Tensor DivOp::forward(const Tensor& t1, const Tensor& t2) {
    if (check_shape_match(t1.shape(), t2.shape())) {
        return Tensor(kernels::div(t1.impl(), t2.impl()));
    }
    BroadcastInfo b_info = construct_broadcast_info(t1.impl(), t2.impl());
    return Tensor(kernels::div_broadcast(t1.impl(), t2.impl(), b_info));
}
void DivOp::backward() {
    BroadcastInfo b_info = construct_broadcast_info(out->grad->impl(), b);
    std::shared_ptr<TensorImpl> divb = kernels::div_broadcast(out->grad->impl(), b, b_info);
    if (a->requires_grad) {
        kernels::add_inplace(a->grad->impl(), reduce_to_shape(divb, a->shape));
    }
    if (b->requires_grad) {
        std::shared_ptr<TensorImpl> temp = kernels::power(b, 2);
        b_info = construct_broadcast_info(a, temp);
        temp = kernels::div_broadcast(a, temp, b_info);
        b_info = construct_broadcast_info(temp, out->grad->impl());
        temp = kernels::mul_broadcast(temp, out->grad->impl(), b_info);
        kernels::sub_inplace(b->grad->impl(), reduce_to_shape(temp, b->shape));
    }
}
std::vector<std::shared_ptr<const TensorImpl>> DivOp::inputs() {
    return {a, b};
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
    }
}
std::vector<std::shared_ptr<const TensorImpl>> ScalarMulOp::inputs() {
    return {a};
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
    }
    if (b->requires_grad) {
        kernels::add_inplace(b->grad->impl(), kernels::matmul(kernels::transpose(a), out->grad->impl()));
    }
}
std::vector<std::shared_ptr<const TensorImpl>> MatMulOp::inputs() {
    return {a, b};
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
    }
    if (b->requires_grad) {
        kernels::add_inplace(b->grad->impl(), col_to_1d(kernels::matmul(kernels::transpose(a), out->grad->impl())));
    }
}
std::vector<std::shared_ptr<const TensorImpl>> MatVecOp::inputs() {
    return {a, b};
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
    }
    if (b->requires_grad) {
        kernels::add_inplace(b->grad->impl(), kernels::scalar_mul(out->grad->impl()->data[0], a));
    }
}
std::vector<std::shared_ptr<const TensorImpl>> DotOp::inputs() {
    return {a, b};
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
std::vector<std::shared_ptr<const TensorImpl>> TransposeOp::inputs() {
    return {a};
}

PowerOp::PowerOp(const Tensor& t1, float x, const Tensor& t2): x(x) {
    a = t1.impl();
    out = t2.impl();
}
Tensor PowerOp::forward(const Tensor& t1, float x) {
    return Tensor(kernels::power(t1.impl(), x));
}
void PowerOp::backward() {
    if (a->requires_grad) {
        kernels::add_inplace(a->grad->impl(), kernels::mul(kernels::scalar_mul(x, kernels::power(a, x-1)), out->grad->impl()));
    }
}
std::vector<std::shared_ptr<const TensorImpl>> PowerOp::inputs() {
    return {a};
}

ExpOp::ExpOp(const Tensor& t1, const Tensor& t2) {
    a = t1.impl();
    out = t2.impl();
}
Tensor ExpOp::forward(const Tensor& t1) {
    return Tensor(kernels::exp(t1.impl()));
}
void ExpOp::backward() {
    if (a->requires_grad) {
        kernels::add_inplace(a->grad->impl(), kernels::mul(out, out->grad->impl()));
    }
}
std::vector<std::shared_ptr<const TensorImpl>> ExpOp::inputs() {
    return {a};
}

LogOp::LogOp(const Tensor& t1, const Tensor& t2) {
    a = t1.impl();
    out = t2.impl();
}
Tensor LogOp::forward(const Tensor& t1) {
    return Tensor(kernels::log(t1.impl()));
}
void LogOp::backward() {
    if (a->requires_grad) {
        kernels::add_inplace(a->grad->impl(), kernels::div(out->grad->impl(), a));
    }
}
std::vector<std::shared_ptr<const TensorImpl>> LogOp::inputs() {
    return {a};
}
