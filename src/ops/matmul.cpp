#include "ops/matmul.h"

// Tensor multiplication: (m, n)(n, l)
Tensor matmul(const Tensor& a, const Tensor& b) {
    // TODO: nD matmul

    std::vector<size_t> a_shape = a.shape();
    std::vector<size_t> b_shape = b.shape();
    if (b_shape.size() != 2 || b_shape.size() != 2) {
        throw std::runtime_error("Matrix multiplication only supported for 2D tensors");
    }
    if (a_shape[a_shape.size()-1] != b_shape[0]) {
        throw std::runtime_error("Cannot matrix multiply tensors with shapes " + a.shape_string() + " and " + b.shape_string());
    }
    Tensor res(0.f, {a_shape[0], b_shape[1]});
    for (size_t i = 0; i < a_shape[0]; ++i) {
        for (size_t j = 0; j < b_shape[1]; ++j) {
            for (size_t k = 0; k < a_shape[1]; ++k) {
                res({i, j}) += a({i, k}) * b({k, j});
            }
        }
    }
    return res;
}

// Tensor multiplication (m, n)(n) or (m, n)(n, 1)
Tensor matvec(const Tensor& a, const Tensor& b) {
    // Expect 2D Tensor and 1D Tensor or 2D column Tensor
    std::vector<size_t> a_shape = a.shape();
    std::vector<size_t> b_shape = b.shape();
    if (a_shape.size() != 2 || (b_shape.size() != 1 && !(b_shape.size() == 2 && b_shape[1] == 1))) {
        throw std::runtime_error("Matrix vector multiplication expects 2D Tensor with 1D Tensor or 2D column Tensor, instead got " + a.shape_string() + " and " + b.shape_string());
    }
    if (a_shape[1] != b.data_raw().size()) {
        throw std::runtime_error("Cannot matrix multiply tensors with shapes " + a.shape_string() + " and " + b.shape_string());
    }
    Tensor res(0.f, {a_shape[0], 1});

    const std::vector<float>& b_data = b.data_raw();
    std::vector<float>& res_data = res.data_raw();
    for (size_t i = 0; i < a_shape[0]; ++i) {
        for (size_t j = 0; j < a_shape[1]; ++j) {
            res_data[i] += a({i, j})*b_data[j];
        }
    }
    return res;
}

Tensor dot(const Tensor& a, const Tensor& b) {
    std::vector<size_t> a_shape = a.shape();
    std::vector<size_t> b_shape = b.shape();
    float res = 0.f;
    if (a_shape.size() == 0 && b_shape.size() == 0) {
        return Tensor({res}, {1});
    }
    const std::vector<float>& a_data = a.data_raw();
    const std::vector<float>& b_data = b.data_raw();
    // a and b 1D tensors
    if (a_shape.size() == 1 && b_shape.size() == 1 && a_shape[0] == b_shape[0]) {
        for (size_t i = 0; i < a_shape[0]; ++i) {
            res += a_data[i] * b_data[i];
        }
        return Tensor({res}, {1});
    }
    // a and b 2D tensors
    if (    (a_shape.size() == 2 && b_shape.size() == 2) &&
            ((a_shape[0] == 1 && b_shape[0] == 1 && a_shape[1] == b_shape[1]) ||
             (a_shape[1] == 1 && b_shape[1] == 1 && a_shape[0] == b_shape[0]))
        ) {
        for (size_t i = 0; i < a_data.size(); ++i) {
            res += a_data[i] * b_data[i];
        }
        return Tensor({res}, {1});
    }
    throw std::runtime_error("Cannot dot product tensors with shapes " + a.shape_string() + " and " + b.shape_string());
}
