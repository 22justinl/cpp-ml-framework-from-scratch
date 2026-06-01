#include "kernels/math_ops.h"

#include "utils/tensor_utils.h"

#include <numbers>

namespace kernels {

// operate on a in place
void add_inplace(std::shared_ptr<TensorImpl> a, std::shared_ptr<const TensorImpl> b) {
    if (!check_tensorimpl_shape_match(a, b)) {
        throw std::runtime_error("Tensor dimension mismatch: " + shape_to_string(a->shape)+ " and " + shape_to_string(b->shape));
    }
    std::vector<float>& a_data = a->data;
    const std::vector<float>& b_data = b->data;

    for (size_t i = 0; i < a_data.size(); ++i) {
        a_data[i] += b_data[i];
    }
}

// operate on a in place
void sub_inplace(std::shared_ptr<TensorImpl> a, std::shared_ptr<const TensorImpl> b) {
    if (!check_tensorimpl_shape_match(a, b)) {
        throw std::runtime_error("Tensor dimension mismatch: " + shape_to_string(a->shape)+ " and " + shape_to_string(b->shape));
    }
    std::vector<float>& a_data = a->data;
    const std::vector<float>& b_data = b->data;

    for (size_t i = 0; i < a_data.size(); ++i) {
        a_data[i] -= b_data[i];
    }
}

void zero_inplace(std::shared_ptr<TensorImpl> a) {
    std::vector<float>& a_data = a->data;
    for (size_t i = 0; i < a_data.size(); ++i) {
        a_data[i] = 0;
    }
}

std::shared_ptr<TensorImpl> add(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b) {
    if (!check_tensorimpl_shape_match(a, b)) {
        throw std::runtime_error("Tensor dimension mismatch: " + shape_to_string(a->shape)+ " and " + shape_to_string(b->shape));
    }
    std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector<float>(a->data.size(), 0.f), a->shape, a->strides, a->requires_grad || b->requires_grad);
    const std::vector<float>& a_data = a->data;
    const std::vector<float>& b_data = b->data;
    std::vector<float>& res_data = res->data;

    for (size_t i = 0; i < a_data.size(); ++i) {
        res_data[i] = a_data[i] + b_data[i];
    }
    return res;
}
std::shared_ptr<TensorImpl> sub(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b) {
    if (b->data.size() == 1) {
        // HACK: support subtracting scalar tensor, remove when broadcasting implemented
        std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector<float>(a->data.size(), 0.f), a->shape, a->strides, a->requires_grad || b->requires_grad);
        const std::vector<float>& a_data = a->data;
        float s = b->data[0];
        std::vector<float>& res_data = res->data;

        for (size_t i = 0; i < a_data.size(); ++i) {
            res_data[i] = a_data[i] - s;
        }
        return res;
    }
    if (!check_tensorimpl_shape_match(a, b)) {
        throw std::runtime_error("Tensor dimension mismatch: " + shape_to_string(a->shape)+ " and " + shape_to_string(b->shape));
    }
    std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector<float>(a->data.size(), 0.f), a->shape, a->strides, a->requires_grad || b->requires_grad);
    const std::vector<float>& a_data = a->data;
    const std::vector<float>& b_data = b->data;
    std::vector<float>& res_data = res->data;

    for (size_t i = 0; i < a_data.size(); ++i) {
        res_data[i] = a_data[i] - b_data[i];
    }
    return res;
}
std::shared_ptr<TensorImpl> mul(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b) {
    if (!check_tensorimpl_shape_match(a, b)) {
        throw std::runtime_error("Tensor dimension mismatch: " + shape_to_string(a->shape)+ " and " + shape_to_string(b->shape));
    }
    std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector<float>(a->data.size(), 0.f), a->shape, a->strides, a->requires_grad || b->requires_grad);
    const std::vector<float>& a_data = a->data;
    const std::vector<float>& b_data = b->data;
    std::vector<float>& res_data = res->data;

    for (size_t i = 0; i < a_data.size(); ++i) {
        res_data[i] = a_data[i] * b_data[i];
    }
    return res;
}
std::shared_ptr<TensorImpl> div(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b) {
    if (!check_tensorimpl_shape_match(a, b)) {
        throw std::runtime_error("Tensor dimension mismatch: " + shape_to_string(a->shape)+ " and " + shape_to_string(b->shape));
    }
    std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector<float>(a->data.size(), 0.f), a->shape, a->strides, a->requires_grad || b->requires_grad);
    const std::vector<float>& a_data = a->data;
    const std::vector<float>& b_data = b->data;
    std::vector<float>& res_data = res->data;

    for (size_t i = 0; i < a_data.size(); ++i) {
        if (!b_data[i]) {
            throw std::runtime_error("Divide by zero error");
        }
        res_data[i] = a_data[i] / b_data[i];
    }
    return res;
}
std::shared_ptr<TensorImpl> div(float f, std::shared_ptr<const TensorImpl> a) {
    std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector<float>(a->data.size(), f), a->shape, a->strides, a->requires_grad);
    const std::vector<float>& a_data = a->data;
    std::vector<float>& res_data = res->data;

    for (size_t i = 0; i < a_data.size(); ++i) {
        if (!a_data[i]) {
            throw std::runtime_error("Divide by zero error");
        }
        res_data[i] /= a_data[i];
    }
    return res;
}
std::shared_ptr<TensorImpl> scalar_mul(float a, std::shared_ptr<const TensorImpl> b) {
    std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector<float>(b->data.size(), 0.f), b->shape, b->strides, b->requires_grad);
    for (size_t i = 0; i < b->data.size(); ++i) {
        res->data[i] = a * b->data[i];
    }
    return res;
}
std::shared_ptr<TensorImpl> matmul(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b) {
    // TODO: nD matmul
    std::vector<size_t> a_shape = a->shape;
    std::vector<size_t> b_shape = b->shape;
    if (a_shape[a_shape.size()-1] != b_shape[0]) {
        throw std::runtime_error("Cannot matrix multiply tensors with shapes " + shape_to_string(a_shape) + " and " + shape_to_string(b_shape));
    }
    if (b_shape.size() != 2 || b_shape.size() != 2) {
        throw std::runtime_error("Matrix multiplication only supported for 2D tensors");
    }
    std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector<float>(a_shape[0]*b_shape[1], 0.f), {a_shape[0], b_shape[1]}, calculate_strides({a_shape[0], b_shape[1]}), a->requires_grad || b->requires_grad);
    for (size_t i = 0; i < a_shape[0]; ++i) {
        for (size_t j = 0; j < b_shape[1]; ++j) {
            for (size_t k = 0; k < a_shape[1]; ++k) {
                res->data[calculate_offset(res, {i, j})] += a->data[calculate_offset(a, {i, k})] * b->data[calculate_offset(b, {k, j})];
            }
        }
    }
    return res;
}
std::shared_ptr<TensorImpl> matvec(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b) {
    std::vector<size_t> t1_shape = a->shape;
    std::vector<size_t> t2_shape = b->shape;
    if (t1_shape.size() != 2 || (t2_shape.size() != 1)) {
        throw std::runtime_error("Matrix vector multiplication expects 2D Tensor with 1D Tensor, instead got " + shape_to_string(a->shape) + " and " + shape_to_string(b->shape));
    }
    if (t1_shape[1] != b->data.size()) {
        throw std::runtime_error("Cannot matrix multiply tensors with shapes " + shape_to_string(a->shape) + " and " + shape_to_string(b->shape));
    }
    std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector<float>(t1_shape[0], 0.f), {t1_shape[0], 1}, {1, 1}, a->requires_grad || b->requires_grad);
    const std::vector<float>& t2_data = b->data;
    std::vector<float>& res_data = res->data;
    for (size_t i = 0; i < t1_shape[0]; ++i) {
        for (size_t j = 0; j < t1_shape[1]; ++j) {
            res_data[i] += a->data[calculate_offset(a, {i,j})]*t2_data[j];
        }
    }
    return res;
}
std::shared_ptr<TensorImpl> dot(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b) {
    std::vector<size_t> t1_shape = a->shape;
    std::vector<size_t> t2_shape = b->shape;
    std::shared_ptr<TensorImpl> res = create_tensorimpl({0.f}, {1}, {1}, a->requires_grad || b->requires_grad);
    if (t1_shape.size() == 0 && t2_shape.size() == 0) {
        return res;
    }
    const std::vector<float>& a_data = a->data;
    const std::vector<float>& b_data = b->data;
    float& res_data = res->data[0];
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
std::shared_ptr<TensorImpl> transpose(std::shared_ptr<const TensorImpl> a) {
    if (a->shape.size() > 2) {
        throw std::runtime_error("Transpose supported for tensors up to 2 dimensions");
    }
    if (a->shape.size() == 0) {
        return create_tensorimpl(std::vector<float>(), {}, {}, a->requires_grad);;
    }
    if (a->shape.size() == 1) {
        return create_tensorimpl(a->data, {1, a->shape[0]}, {a->shape[0], 1}, a->requires_grad);
    }
    std::vector<size_t> new_shape({a->shape[1], a->shape[0]});
    std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector<float>(a->data.size(), 0.f), new_shape, {new_shape[1], 1}, a->requires_grad);
    for (size_t i = 0; i < a->shape[0]; ++i) {
        for (size_t j = 0; j < a->shape[1]; ++j) {
            res->data[calculate_offset(res, {j, i})] = a->data[calculate_offset(a, {i, j})];
        }
    }
    return res;
}

std::shared_ptr<TensorImpl> power(std::shared_ptr<const TensorImpl> a, float x) {
    std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector(a->data), a->shape, a->strides, a->requires_grad);
    for (size_t i = 0; i < a->data.size(); ++i) {
        res->data[i] = std::pow(res->data[i], x);
    }
    return res;
}
std::shared_ptr<TensorImpl> exp(std::shared_ptr<const TensorImpl> a) {
    std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector(a->data), a->shape, a->strides, a->requires_grad);
    for (size_t i = 0; i < a->data.size(); ++i) {
        res->data[i] = std::pow(std::numbers::e_v<float>, res->data[i]);
    }
    return res;
}
std::shared_ptr<TensorImpl> log(std::shared_ptr<const TensorImpl> a) {
    std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector(a->data), a->shape, a->strides, a->requires_grad);
    for (size_t i = 0; i < a->data.size(); ++i) {
        res->data[i] = std::log(res->data[i]);
    }
    return res;
}

}

// TODO: replace later with reshaping?
std::shared_ptr<TensorImpl> col_to_1d(std::shared_ptr<TensorImpl> a) {
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
