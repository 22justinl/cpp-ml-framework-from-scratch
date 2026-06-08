#include "kernels/math_ops.h"

#include <numbers>

using std::shared_ptr;
using std::make_shared;

namespace kernels {

void add_inplace(shared_ptr<TensorImpl> a, shared_ptr<const TensorImpl> b) {
    elementwise_binary_op_inplace(a, b, [](float a, float b) { return a + b; });
}
void sub_inplace(shared_ptr<TensorImpl> a, shared_ptr<const TensorImpl> b) {
    elementwise_binary_op_inplace(a, b, [](float a, float b) { return a - b; });
}
void add_inplace_broadcast(shared_ptr<TensorImpl> a, shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info) {
    elementwise_binary_op_inplace_broadcast(a, b, [](float a, float b) { return a + b; }, b_info);
}
void sub_inplace_broadcast(shared_ptr<TensorImpl> a, shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info) {
    elementwise_binary_op_inplace_broadcast(a, b, [](float a, float b) { return a - b; }, b_info);
}

void zero_inplace(shared_ptr<TensorImpl> a) {
    std::vector<float>& a_data = a->storage->data;
    for (size_t i = 0; i < a_data.size(); ++i) {
        a_data[i] = 0;
    }
}

shared_ptr<TensorImpl> add(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b) {
    return elementwise_binary_op(a, b, [](float a, float b){ return a + b; });
}
shared_ptr<TensorImpl> sub(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b) {
    return elementwise_binary_op(a, b, [](float a, float b){ return a - b; });
}
shared_ptr<TensorImpl> mul(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b) {
    return elementwise_binary_op(a, b, [](float a, float b){ return a * b; });
}
shared_ptr<TensorImpl> div(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b) {
    return elementwise_binary_op(a, b, [](float a, float b){ if (b == 0) { throw std::runtime_error("Divide by 0"); } return a / b; });
}
shared_ptr<TensorImpl> add_broadcast(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info) {
    return elementwise_binary_op_broadcast(a, b, [](float a, float b){ return a + b; }, b_info);
}
shared_ptr<TensorImpl> sub_broadcast(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info) {
    return elementwise_binary_op_broadcast(a, b, [](float a, float b){ return a - b; }, b_info);
}
shared_ptr<TensorImpl> mul_broadcast(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info) {
    return elementwise_binary_op_broadcast(a, b, [](float a, float b){ return a * b; }, b_info);
}
shared_ptr<TensorImpl> div_broadcast(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info) {
    return elementwise_binary_op_broadcast(a, b, [](float a, float b){ if (b == 0) { throw std::runtime_error("Divide by 0"); } return a / b; }, b_info);
}
shared_ptr<TensorImpl> div(float f, shared_ptr<const TensorImpl> a) {
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(f, a->shape, a->strides, a->requires_grad);
    const std::vector<float>& a_data = a->storage->data;
    std::vector<float>& res_data = res->storage->data;

    for (size_t i = 0; i < a_data.size(); ++i) {
        if (!a_data[i]) {
            throw std::runtime_error("Divide by zero error");
        }
        res_data[i] /= a_data[i];
    }
    return res;
}
shared_ptr<TensorImpl> scalar_mul(float a, shared_ptr<const TensorImpl> b) {
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, b->shape, b->strides, b->requires_grad);
    for (size_t i = 0; i < b->storage->data.size(); ++i) {
        res->storage->data[i] = a * b->storage->data[i];
    }
    return res;
}
shared_ptr<TensorImpl> matmul(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b) {
    // TODO: broadcasting with matmul (different rules from elementwise operations)
    // TODO: nD matmul
    std::vector<size_t> a_shape = a->shape;
    std::vector<size_t> b_shape = b->shape;
    if (a_shape[a_shape.size()-1] != b_shape[0]) {
        throw std::runtime_error("Cannot matrix multiply tensors with shapes " + shape_to_string(a_shape) + " and " + shape_to_string(b_shape));
    }
    if (b_shape.size() != 2 || b_shape.size() != 2) {
        throw std::runtime_error("Matrix multiplication only supported for 2D tensors");
    }
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, std::vector<size_t>({a_shape[0], b_shape[1]}), a->requires_grad || b->requires_grad);
    for (size_t i = 0; i < a_shape[0]; ++i) {
        for (size_t j = 0; j < b_shape[1]; ++j) {
            for (size_t k = 0; k < a_shape[1]; ++k) {
                res->storage->data[calculate_offset(res, {i, j})] += a->storage->data[calculate_offset(a, {i, k})] * b->storage->data[calculate_offset(b, {k, j})];
            }
        }
    }
    return res;
}
shared_ptr<TensorImpl> matvec(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b) {
    std::vector<size_t> t1_shape = a->shape;
    std::vector<size_t> t2_shape = b->shape;
    if (t1_shape.size() != 2 || (t2_shape.size() != 1)) {
        throw std::runtime_error("Matrix vector multiplication expects 2D Tensor with 1D Tensor, instead got " + shape_to_string(a->shape) + " and " + shape_to_string(b->shape));
    }
    if (t1_shape[1] != b->storage->data.size()) {
        throw std::runtime_error("Cannot matrix multiply tensors with shapes " + shape_to_string(a->shape) + " and " + shape_to_string(b->shape));
    }
    shared_ptr res = make_shared<TensorImpl>(0, std::vector<size_t>({t1_shape[0], 1}), a->requires_grad || b->requires_grad);
    const std::vector<float>& t2_data = b->storage->data;
    std::vector<float>& res_data = res->storage->data;
    for (size_t i = 0; i < t1_shape[0]; ++i) {
        for (size_t j = 0; j < t1_shape[1]; ++j) {
            res_data[i] += a->storage->data[calculate_offset(a, {i,j})]*t2_data[j];
        }
    }
    return res;
}
shared_ptr<TensorImpl> dot(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b) {
    std::vector<size_t> t1_shape = a->shape;
    std::vector<size_t> t2_shape = b->shape;
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, std::vector<size_t>({1}), a->requires_grad);
    if (t1_shape.size() == 0 && t2_shape.size() == 0) {
        return res;
    }
    const std::vector<float>& a_data = a->storage->data;
    const std::vector<float>& b_data = b->storage->data;
    float& res_data = res->storage->data[0];
    // a and b 1D tensors
    if (t1_shape.size() == 1 && t2_shape.size() == 1 && t1_shape[0] == t2_shape[0]) {
        for (size_t i = 0; i < t1_shape[0]; ++i) {
            res_data += a_data[i] * b_data[i];
        }
        return res;
    }
    // NOTE: PyTorch only supports dot product of 1D tensors, using same behavior here
    // // a and b 2D tensors
    // if (    (t1_shape.size() == 2 && t2_shape.size() == 2) &&
    //         ((t1_shape[0] == 1 && t2_shape[0] == 1 && t1_shape[1] == t2_shape[1]) ||
    //          (t1_shape[1] == 1 && t2_shape[1] == 1 && t1_shape[0] == t2_shape[0]))
    //     ) {
    //     for (size_t i = 0; i < a_data.size(); ++i) {
    //         res_data += a_data[i] * b_data[i];
    //     }
    //     return res;
    // }
    throw std::runtime_error("Cannot dot product tensors with shapes " + shape_to_string(a->shape) + " and " + shape_to_string(b->shape));
}

shared_ptr<TensorImpl> power(shared_ptr<const TensorImpl> a, float x) {
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(a->storage->data, a->shape, a->strides, a->requires_grad);
    for (size_t i = 0; i < a->storage->data.size(); ++i) {
        res->storage->data[i] = std::pow(res->storage->data[i], x);
    }
    return res;
}
shared_ptr<TensorImpl> exp(shared_ptr<const TensorImpl> a) {
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(a->storage->data, a->shape, a->strides, a->requires_grad);
    for (size_t i = 0; i < a->storage->data.size(); ++i) {
        res->storage->data[i] = std::pow(std::numbers::e_v<float>, res->storage->data[i]);
    }
    return res;
}
shared_ptr<TensorImpl> log(shared_ptr<const TensorImpl> a) {
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(a->storage->data, a->shape, a->strides, a->requires_grad);
    for (size_t i = 0; i < a->storage->data.size(); ++i) {
        res->storage->data[i] = std::log(res->storage->data[i]);
    }
    return res;
}

}

// TODO: replace later with reshaping?
shared_ptr<TensorImpl> col_to_vec(shared_ptr<TensorImpl> a) {
    if (a->shape.size() != 2 || a->shape[1] != 1) {
        throw std::runtime_error("Only 2D column Tensors can be converted to 1D Tensor");
    }
    a->shape = {a->shape[0]};
    a->strides = calculate_strides(a->shape);
    if (a->grad) {
        a->grad->impl()->shape = a->shape;
        a->grad->impl()->strides = a->strides;
    }
    return a;
}
shared_ptr<TensorImpl> vec_to_col(shared_ptr<TensorImpl> a) {
    if (a->shape.size() != 1) {
        throw std::runtime_error("Only 1D Tensors can be converted to column Tensor");
    }
    a->shape = {a->shape[0], 1};
    a->strides = calculate_strides(a->shape);
    if (a->grad) {
        a->grad->impl()->shape = a->shape;
        a->grad->impl()->strides = a->strides;
    }
    return a;
}
