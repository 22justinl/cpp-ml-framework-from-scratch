#pragma once
#include "core/tensor.h"
#include "core/broadcast.h"
#include "utils/tensor_utils.h"

namespace kernels {
template <typename Op>
std::shared_ptr<TensorImpl> elementwise_binary_op(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b, Op op) {
    if (!check_shape_match(a->shape, b->shape)) {
        throw std::runtime_error("Tensor shape mismatch");
    }
    std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector<float>(a->data.size(), 0), a->shape, a->strides, a->requires_grad || b->requires_grad);
    for (size_t i = 0; i < res->data.size(); ++i) {
        res->data[i] = op(a->data[i], b->data[i]);
    }
    return res;
}
template <typename Op>
std::shared_ptr<TensorImpl> elementwise_binary_op_broadcast(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b, Op op, const BroadcastInfo& b_info) {
    const std::vector<size_t>& out_shape = b_info.out_shape;
    const std::vector<size_t>& a_strides = b_info.a_strides;
    const std::vector<size_t>& b_strides = b_info.b_strides;
    const std::vector<size_t>& out_strides = b_info.out_strides;
    std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector<float>(calculate_n_el(out_shape), 0), out_shape, out_strides, a->requires_grad || b->requires_grad);
    std::vector<size_t> idx(out_shape.size(), 0);
    for (size_t i = 0; i < res->data.size(); ++i) {
        map_offset_to_idx(i, out_strides, idx);
        res->data[i] = op(a->data[map_idx_to_offset(idx, a_strides)], b->data[map_idx_to_offset(idx, b_strides)]);
    }
    return res;
}
template <typename Op>
void elementwise_binary_op_inplace(std::shared_ptr<TensorImpl> a, std::shared_ptr<const TensorImpl> b, Op op) {
    if (!check_shape_match(a->shape, b->shape)) {
        throw std::runtime_error("Tensor shape mismatch");
    }
    std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector<float>(a->data.size(), 0), a->shape, a->strides, a->requires_grad || b->requires_grad);
    for (size_t i = 0; i < res->data.size(); ++i) {
        a->data[i] = op(a->data[i], b->data[i]);
    }
}
template <typename Op>
void elementwise_binary_op_inplace_broadcast(std::shared_ptr<TensorImpl> a, std::shared_ptr<const TensorImpl> b, Op op, const BroadcastInfo& b_info) {
    const std::vector<size_t>& b_strides = b_info.b_strides;
    std::vector<size_t> idx(a->shape.size(), 0);
    for (size_t i = 0; i < a->data.size(); ++i) {
        map_offset_to_idx(i, a->shape, idx);
        a->data[i] = op(a->data[i], b->data[map_idx_to_offset(idx, b_strides)]);
    }
}
void add_inplace(std::shared_ptr<TensorImpl> a, std::shared_ptr<const TensorImpl> b);
void sub_inplace(std::shared_ptr<TensorImpl> a, std::shared_ptr<const TensorImpl> b);
void add_inplace_broadcast(std::shared_ptr<TensorImpl> a, std::shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info);
void sub_inplace_broadcast(std::shared_ptr<TensorImpl> a, std::shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info);
void zero_inplace(std::shared_ptr<TensorImpl> a);
std::shared_ptr<TensorImpl> add(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b);
std::shared_ptr<TensorImpl> sub(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b);
std::shared_ptr<TensorImpl> mul(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b);
std::shared_ptr<TensorImpl> div(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b);
std::shared_ptr<TensorImpl> add_broadcast(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info);
std::shared_ptr<TensorImpl> sub_broadcast(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info);
std::shared_ptr<TensorImpl> mul_broadcast(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info);
std::shared_ptr<TensorImpl> div_broadcast(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info);
std::shared_ptr<TensorImpl> div(float f, std::shared_ptr<const TensorImpl> a);
std::shared_ptr<TensorImpl> scalar_mul(float a, std::shared_ptr<const TensorImpl> b);
std::shared_ptr<TensorImpl> matmul(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b);
std::shared_ptr<TensorImpl> matvec(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b);
std::shared_ptr<TensorImpl> dot(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b);
std::shared_ptr<TensorImpl> transpose(std::shared_ptr<const TensorImpl> a);

std::shared_ptr<TensorImpl> power(std::shared_ptr<const TensorImpl> a, float x);
std::shared_ptr<TensorImpl> exp(std::shared_ptr<const TensorImpl> a);
std::shared_ptr<TensorImpl> log(std::shared_ptr<const TensorImpl> a); // natural log

}
// helpers
std::shared_ptr<TensorImpl> col_to_vec(std::shared_ptr<TensorImpl> a);
std::shared_ptr<TensorImpl> vec_to_col(std::shared_ptr<TensorImpl> a);
