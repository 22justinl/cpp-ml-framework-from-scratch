#include "kernels/math_ops.h"
#include "utils/tensor_utils.h"

namespace kernels {

// operate on a in place
void add_inplace(std::shared_ptr<TensorImpl> a, std::shared_ptr<const TensorImpl> b) {
    if (!check_tensorimpl_shape_match(a, b)) {
        throw std::runtime_error("TensorImpl dimension mismatch (mathop kernel)");
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
        throw std::runtime_error("TensorImpl dimension mismatch (mathop kernel)");
    }
    std::vector<float>& a_data = a->data;
    const std::vector<float>& b_data = b->data;

    for (size_t i = 0; i < a_data.size(); ++i) {
        a_data[i] -= b_data[i];
    }
}

std::shared_ptr<TensorImpl> add(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b) {
    if (!check_tensorimpl_shape_match(a, b)) {
        throw std::runtime_error("TensorImpl dimension mismatch (mathop kernel)");
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
    if (!check_tensorimpl_shape_match(a, b)) {
        throw std::runtime_error("TensorImpl dimension mismatch (mathop kernel)");
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
        throw std::runtime_error("TensorImpl dimension mismatch (mathop kernel)");
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
        throw std::runtime_error("TensorImpl dimension mismatch (mathop kernel)");
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
std::shared_ptr<TensorImpl> matmul(std::shared_ptr<const TensorImpl> a, std::shared_ptr<const TensorImpl> b) {
    // TODO: nD matmul
    std::vector<size_t> a_shape = a->shape;
    std::vector<size_t> b_shape = b->shape;
    if (b_shape.size() != 2 || b_shape.size() != 2) {
        throw std::runtime_error("Matrix multiplication only supported for 2D tensors");
    }
    std::shared_ptr<TensorImpl> res = create_tensorimpl(std::vector<float>(a_shape[0]*b_shape[1], 0.f), {a_shape[0], b_shape[1]}, a->strides, a->requires_grad || b->requires_grad);
    for (size_t i = 0; i < a_shape[0]; ++i) {
        for (size_t j = 0; j < b_shape[1]; ++j) {
            for (size_t k = 0; k < a_shape[1]; ++k) {
                res->data[calculate_offset(res, {i, j})] += a->data[calculate_offset(a, {i, k})] * b->data[calculate_offset(b, {k, j})];
            }
        }
    }
    return res;
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

}
