#include "ops/math_ops.h"

#include "autograd/math_ops.h"
#include "utils/op_utils.h"

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
    if (t1.requires_grad()) {
        std::shared_ptr<ScalarMulOp> op = std::make_shared<ScalarMulOp>(f, t1, t2);
        t2.set_grad_fn(op);
    }
    return t2;
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
    return monop_helper<TransposeOp>(t1);
}
Tensor power(const Tensor& t1, float f) {
    Tensor t2 = PowerOp::forward(t1, f);
    if (t1.requires_grad()) {
        std::shared_ptr<PowerOp> op = std::make_shared<PowerOp>(t1, f, t2);
        t2.set_grad_fn(op);
    }
    return t2;
}
Tensor exp(const Tensor& t1) {
    return monop_helper<ExpOp>(t1);
}
Tensor log_e(const Tensor& t1) {
    return monop_helper<LogOp>(t1);
}
