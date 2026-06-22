#include "loss.h"

#include "ops/math_ops.h"
#include "ops/reduction_ops.h"

Tensor mse_loss(const Tensor& pred, const Tensor& target) {
    return sum(power(target-pred, 2), 1)/float(pred.shape()[1]);
}

Tensor cross_entropy_loss(const Tensor& pred, const Tensor& target) {
    Tensor m = max(pred);
    Tensor shifted = pred-m;
    Tensor last_term = log_e(sum(exp(shifted), 1));
    Tensor t3 = -sum(target * (shifted-last_term), 1);
    return t3;
}
