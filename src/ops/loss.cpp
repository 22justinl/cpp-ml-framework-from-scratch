#include "loss.h"

#include "autograd/math_ops.h"
#include "ops/math_ops.h"
#include "ops/reduction_ops.h"

Tensor mse_loss(const Tensor& pred, const Tensor& target) {
    Tensor s = sum((target-pred)*(target-pred));
    Tensor t3 = s/pred.shape()[0];
    if (pred.requires_grad() || target.requires_grad()) {
        std::shared_ptr<ScalarMulOp> op = std::make_shared<ScalarMulOp>(1.0/pred.shape()[0], s, t3);
        t3.set_grad_fn(op);
    }
    return t3;
}

Tensor cross_entropy_loss(const Tensor& input, const Tensor& target) {
    // Tensor m = max(input);
    // Tensor shifted = input-m;
    // Tensor last_term = log_e(sum(exp(shifted)));
    // Tensor s = sum(target * (shifted-last_term));
    // Tensor t3 = -s;

    Tensor last_term = log_e(sum(exp(input)));
    Tensor s = sum(target * (input-last_term));
    Tensor t3 = -s;
    if (input.requires_grad() || target.requires_grad()) {
        std::shared_ptr<ScalarMulOp> op = std::make_shared<ScalarMulOp>(-1.0, s, t3);
        t3.set_grad_fn(op);
    }
    return t3;
}
