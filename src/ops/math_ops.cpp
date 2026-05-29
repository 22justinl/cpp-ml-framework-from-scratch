#include "ops/math_ops.h"
#include "autograd/math_ops.h"

template <class BinOp>
Tensor binop_helper(const Tensor& t1, const Tensor& t2) {
    Tensor t3 = BinOp::forward(t1, t2);
    if (t1.requires_grad() || t2.requires_grad()) {
        std::shared_ptr<BinOp> op = std::make_shared<BinOp>(t1, t2, t3);
        t3.set_grad_fn(op);
    }
    return t3;
}

Tensor add(const Tensor& t1, const Tensor& t2) {
    return binop_helper<AddOp>(t1, t2);
}
Tensor sub(const Tensor& t1, const Tensor& t2) {
    return binop_helper<SubOp>(t1, t2);
}
Tensor mul(const Tensor& t1, const Tensor& t2) {
    return binop_helper<MulOp>(t1, t2);
}
Tensor div(const Tensor& t1, const Tensor& t2) {
    return binop_helper<DivOp>(t1, t2);
}
Tensor scalar_mul(const Tensor& t1, float f) { return scalar_mul(f, t1); }
Tensor scalar_mul(float f, const Tensor& t1) {
    Tensor t2 = ScalarMulOp::forward(f, t1);
    return binop_helper<DivOp>(t1, t2);
}

// Tensor multiplication: (m, n)(n, l)
Tensor matmul(const Tensor& t1, const Tensor& t2) {
    return binop_helper<MatMulOp>(t1, t2);
}
// Tensor multiplication (m, n)(n) or (m, n)(n, 1)
Tensor matvec(const Tensor& t1, const Tensor& t2) {
    return binop_helper<MatVecOp>(t1, t2);
}
Tensor dot(const Tensor& t1, const Tensor& t2) {
    return binop_helper<DotOp>(t1, t2);
}
Tensor transpose(const Tensor& t1) {
    Tensor t2 = TransposeOp::forward(t1);
    if (t1.requires_grad()) {
        std::shared_ptr<TransposeOp> op = std::make_shared<TransposeOp>(t1, t2);
        t2.set_grad_fn(op);
    }
    return t2;
}
