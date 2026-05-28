#include "ops/math_ops.h"
#include "autograd/math_ops.h"

template <class BinOp>
Tensor binop_helper(const Tensor& a, const Tensor& b) {
    Tensor c = BinOp::forward(a, b);
    if (a.requires_grad() || b.requires_grad()) {
        std::shared_ptr<BinOp> op = std::make_shared<BinOp>(a, b, c);
        c.set_grad_fn(op);
    }
    return c;
}

Tensor add(const Tensor& a, const Tensor& b) {
    return binop_helper<AddOp>(a, b);
}
Tensor sub(const Tensor& a, const Tensor& b) {
    return binop_helper<SubOp>(a, b);
}
Tensor mul(const Tensor& a, const Tensor& b) {
    return binop_helper<MulOp>(a, b);
}
Tensor div(const Tensor& a, const Tensor& b) {
    return binop_helper<DivOp>(a, b);
}

// Tensor multiplication: (m, n)(n, l)
Tensor matmul(const Tensor& a, const Tensor& b) {
    return binop_helper<MatmulOp>(a, b);
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

Tensor transpose(const Tensor& t) {
    if (t.shape().size() > 2) {
        throw std::runtime_error("Transpose supported for tensors up to 2 dimensions");
    }
    if (t.shape().size() == 0) {
        return t;
    }
    if (t.shape().size() == 1) {
        return Tensor(t.data_raw(), {1, t.shape()[0]});
    }
    std::vector<size_t> new_shape({t.shape()[1], t.shape()[0]});
    Tensor res(0.f, new_shape);
    for (size_t i = 0; i < t.shape()[0]; ++i) {
        for (size_t j = 0; j < t.shape()[1]; ++j) {
            res({j, i}) = t({i, j});
        }
    }
    return res;
}
