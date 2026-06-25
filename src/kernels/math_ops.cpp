#include "kernels/math_ops.h"

#include "kernels/tensor_ops.h"
#include "utils/tensor_utils.h"

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
    if (a->shape.size() != 2 || b->shape.size() != 2) {
        throw std::runtime_error("Matrix multiplication only supported for 2D tensors");
    }
    if (a->shape[a->shape.size()-1] != b->shape[0]) {
        throw std::runtime_error("Cannot matrix multiply tensors with shapes " + shape_to_string(a->shape) + " and " + shape_to_string(b->shape));
    }
    shared_ptr<const TensorImpl> a_cont = is_contiguous(a) ? a : kernels::contiguous(a);
    shared_ptr<const TensorImpl> b_t = kernels::transpose(b, 0, 1);
    shared_ptr<const TensorImpl> b_t_cont = is_contiguous(b_t) ? b_t : kernels::contiguous(b_t);
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, std::vector<size_t>({a->shape[0], b->shape[1]}), a->requires_grad || b->requires_grad);

    std::vector<float>& res_data = res->storage->data;
    const std::vector<float>& a_data = a_cont->storage->data;
    const std::vector<float>& b_t_data = b_t_cont->storage->data;

    const std::vector<size_t>& a_shape = a_cont->shape;
    const std::vector<size_t>& b_t_shape = b_t_cont->shape;
    const std::vector<size_t>& res_shape = res->shape;

    const size_t M = res_shape[0];
    const size_t N = res_shape[1];
    const size_t K = a_cont->shape[1];
    const size_t B = 16;
    for (size_t ii = 0; ii < M; ii+=B) {
        for (size_t jj = 0; jj < N; jj+=B) {
            for (size_t kk = 0; kk < K; kk+=B) {
                for (size_t i = ii; i < std::min(ii+B, M); ++i) {
                    const size_t a_offset = a_cont->offset + i * K;
                    for (size_t j = jj; j < std::min(jj+B, N); ++j) {
                        const size_t b_t_offset = b_t_cont->offset + j * K;
                        float s = 0;
                        for (size_t k = kk; k < std::min(kk+B, K); ++k) {
                            s += a_data[a_offset+k]*b_t_data[b_t_offset+k];
                        }
                        res_data[i*N+j] += s;
                    }
                }
            }
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
shared_ptr<TensorImpl> col_to_vec(shared_ptr<const TensorImpl> a) {
    if (a->shape.size() != 2 || a->shape[1] != 1) {
        throw std::runtime_error("Only 2D column Tensors can be converted to 1D Tensor");
    }
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(a->storage, a->shape, a->strides, a->requires_grad);
    res->shape = {res->shape[0]};
    res->strides = calculate_strides(res->shape);
    if (res->grad) {
        res->grad->impl()->shape = res->shape;
        res->grad->impl()->strides = res->strides;
    }
    return res;
}
shared_ptr<TensorImpl> vec_to_col(shared_ptr<const TensorImpl> a) {
    if (a->shape.size() != 1) {
        throw std::runtime_error("Only 1D Tensors can be converted to column Tensor");
    }
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(a->storage, a->shape, a->strides, a->requires_grad);
    res->shape = {res->shape[0], 1};
    res->strides = calculate_strides(res->shape);
    if (res->grad) {
        res->grad->impl()->shape = res->shape;
        res->grad->impl()->strides = res->strides;
    }
    return res;
}
