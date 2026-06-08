#pragma once
#include "core/tensor.h"
#include "core/broadcast.h"
#include "utils/tensor_utils.h"

using std::shared_ptr;
using std::make_shared;

namespace kernels {
template <typename Op>
shared_ptr<TensorImpl> elementwise_binary_op(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b, Op op) {
    if (!check_shape_match(a->shape, b->shape)) {
        throw std::runtime_error("Tensor shape mismatch");
    }
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, a->shape, a->strides, a->requires_grad || b->requires_grad);
    for (size_t i = 0; i < res->storage->data.size(); ++i) {
        res->storage->data[i] = op(a->storage->data[i], b->storage->data[i]);
    }
    return res;
}
template <typename Op>
shared_ptr<TensorImpl> elementwise_binary_op_broadcast(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b, Op op, const BroadcastInfo& b_info) {
    const std::vector<size_t>& out_shape = b_info.out_shape;
    const std::vector<size_t>& a_strides = b_info.a_strides;
    const std::vector<size_t>& b_strides = b_info.b_strides;
    const std::vector<size_t>& out_strides = b_info.out_strides;
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, out_shape, out_strides, a->requires_grad || b->requires_grad);
    std::vector<size_t> idx(out_shape.size(), 0);
    for (size_t i = 0; i < res->storage->data.size(); ++i) {
        map_offset_to_idx(i, out_strides, idx);
        res->storage->data[i] = op(a->storage->data[map_idx_to_offset(idx, a_strides)], b->storage->data[map_idx_to_offset(idx, b_strides)]);
    }
    return res;
}
template <typename Op>
void elementwise_binary_op_inplace(shared_ptr<TensorImpl> a, shared_ptr<const TensorImpl> b, Op op) {
    if (!check_shape_match(a->shape, b->shape)) {
        throw std::runtime_error("Tensor shape mismatch");
    }
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, a->shape, a->strides, a->requires_grad || b->requires_grad);
    for (size_t i = 0; i < res->storage->data.size(); ++i) {
        a->storage->data[i] = op(a->storage->data[i], b->storage->data[i]);
    }
}
template <typename Op>
void elementwise_binary_op_inplace_broadcast(shared_ptr<TensorImpl> a, shared_ptr<const TensorImpl> b, Op op, const BroadcastInfo& b_info) {
    const std::vector<size_t>& b_strides = b_info.b_strides;
    std::vector<size_t> idx(a->shape.size(), 0);
    for (size_t i = 0; i < a->storage->data.size(); ++i) {
        map_offset_to_idx(i, a->shape, idx);
        a->storage->data[i] = op(a->storage->data[i], b->storage->data[map_idx_to_offset(idx, b_strides)]);
    }
}
void add_inplace(shared_ptr<TensorImpl> a, shared_ptr<const TensorImpl> b);
void sub_inplace(shared_ptr<TensorImpl> a, shared_ptr<const TensorImpl> b);
void add_inplace_broadcast(shared_ptr<TensorImpl> a, shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info);
void sub_inplace_broadcast(shared_ptr<TensorImpl> a, shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info);
void zero_inplace(shared_ptr<TensorImpl> a);
shared_ptr<TensorImpl> add(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b);
shared_ptr<TensorImpl> sub(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b);
shared_ptr<TensorImpl> mul(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b);
shared_ptr<TensorImpl> div(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b);
shared_ptr<TensorImpl> add_broadcast(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info);
shared_ptr<TensorImpl> sub_broadcast(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info);
shared_ptr<TensorImpl> mul_broadcast(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info);
shared_ptr<TensorImpl> div_broadcast(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info);
shared_ptr<TensorImpl> div(float f, shared_ptr<const TensorImpl> a);
shared_ptr<TensorImpl> scalar_mul(float a, shared_ptr<const TensorImpl> b);
shared_ptr<TensorImpl> matmul(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b);
shared_ptr<TensorImpl> matvec(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b);
shared_ptr<TensorImpl> dot(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b);

shared_ptr<TensorImpl> power(shared_ptr<const TensorImpl> a, float x);
shared_ptr<TensorImpl> exp(shared_ptr<const TensorImpl> a);
shared_ptr<TensorImpl> log(shared_ptr<const TensorImpl> a); // natural log

}
// helpers
shared_ptr<TensorImpl> col_to_vec(shared_ptr<TensorImpl> a);
shared_ptr<TensorImpl> vec_to_col(shared_ptr<TensorImpl> a);
