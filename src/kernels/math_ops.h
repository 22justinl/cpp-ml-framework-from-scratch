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
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, a->shape, a->requires_grad || b->requires_grad);

    std::vector<size_t> idx(res->shape.size(), 0);
    const std::vector<size_t>& shape = res->shape;

    size_t a_offset = a->offset;
    size_t b_offset = b->offset;
    size_t res_offset = res->offset;

    const std::vector<size_t>& a_strides = a->strides;
    const std::vector<size_t>& b_strides = b->strides;
    const std::vector<size_t>& res_strides = res->strides;

    const std::vector<float>& a_data = a->storage->data;
    const std::vector<float>& b_data = b->storage->data;
    std::vector<float>& res_data = res->storage->data;

    const size_t n_el = res->n_el;
    for (size_t i = 0; i < n_el; ++i) {
        res_data[res_offset] = op(a_data[a_offset], b_data[b_offset]);
        increment_offset_binary_op(idx, shape,
                a_offset, a_strides,
                b_offset, b_strides,
                res_offset, res_strides);
    }
    return res;
}
template <typename Op>
shared_ptr<TensorImpl> elementwise_binary_op_broadcast(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b, Op op, const BroadcastInfo& b_info) {
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, b_info.out_shape, b_info.out_strides, a->requires_grad || b->requires_grad);

    std::vector<size_t> idx(b_info.out_shape.size(), 0);
    const std::vector<size_t>& shape = b_info.out_shape;

    size_t a_offset = a->offset;
    size_t b_offset = b->offset;
    size_t res_offset = res->offset;

    const std::vector<size_t>& a_strides = b_info.a_strides;
    const std::vector<size_t>& b_strides = b_info.b_strides;
    const std::vector<size_t>& res_strides = b_info.out_strides;

    const std::vector<float>& a_data = a->storage->data;
    const std::vector<float>& b_data = b->storage->data;
    std::vector<float>& res_data = res->storage->data;

    const size_t n_el = res->n_el;
    for (size_t i = 0; i < n_el; ++i) {
        res_data[res_offset] = op(a_data[a_offset], b_data[b_offset]);
        increment_offset_binary_op(idx, shape,
                a_offset, a_strides,
                b_offset, b_strides,
                res_offset, res_strides);
    }
    return res;
}
template <typename Op>
void elementwise_binary_op_inplace(shared_ptr<TensorImpl> a, shared_ptr<const TensorImpl> b, Op op) {
    if (!check_shape_match(a->shape, b->shape)) {
        throw std::runtime_error("Tensor shape mismatch");
    }

    std::vector<size_t> idx(a->shape.size(), 0);
    const std::vector<size_t>& shape = a->shape;

    size_t a_offset = a->offset;
    size_t b_offset = b->offset;

    const std::vector<size_t>& a_strides = a->strides;
    const std::vector<size_t>& b_strides = b->strides;

    std::vector<float>& a_data = a->storage->data;
    const std::vector<float>& b_data = b->storage->data;

    const size_t n_el = a->n_el;
    for (size_t i = 0; i < n_el; ++i) {
        a_data[a_offset] = op(a_data[a_offset], b_data[b_offset]);
        increment_offset_unary_op(idx, shape,
                b_offset, b_strides,
                a_offset, a_strides);
    }
}
template <typename Op>
void elementwise_binary_op_inplace_broadcast(shared_ptr<TensorImpl> a, shared_ptr<const TensorImpl> b, Op op, const BroadcastInfo& b_info) {
    std::vector<size_t> idx(b_info.out_shape.size(), 0);
    const std::vector<size_t>& shape = b_info.out_shape;

    size_t a_offset = a->offset;
    size_t b_offset = b->offset;

    const std::vector<size_t>& a_strides = b_info.a_strides;
    const std::vector<size_t>& b_strides = b_info.b_strides;

    std::vector<float>& a_data = a->storage->data;
    const std::vector<float>& b_data = b->storage->data;

    const size_t n_el = a->n_el;
    for (size_t i = 0; i < n_el; ++i) {
        a_data[a_offset] = op(a_data[a_offset], b_data[b_offset]);
        increment_offset_unary_op(idx, shape,
                b_offset, b_strides,
                a_offset, a_strides);
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
shared_ptr<TensorImpl> dot(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b);

shared_ptr<TensorImpl> power(shared_ptr<const TensorImpl> a, float x);
shared_ptr<TensorImpl> exp(shared_ptr<const TensorImpl> a);
shared_ptr<TensorImpl> log(shared_ptr<const TensorImpl> a); // natural log

}
// helpers
shared_ptr<TensorImpl> col_to_vec(shared_ptr<const TensorImpl> a);
shared_ptr<TensorImpl> vec_to_col(shared_ptr<const TensorImpl> a);
