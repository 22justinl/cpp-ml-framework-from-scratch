#include "loss.h"

#include "ops/math_ops.h"
#include "ops/reduction_ops.h"
#include "ops/tensor_ops.h"
#include "utils/tensor_utils.h"

Tensor mse_loss(const Tensor& pred, const Tensor& target) {
    return sum(power(target-pred, 2), 1)/float(pred.shape()[1]);
}

Tensor cross_entropy_loss(Tensor pred, Tensor target) {
    if (!check_shape_match(pred.shape(), target.shape())) {
        if (target.shape().size() != 1 || target.shape()[0] != pred.shape()[0]) {
            throw std::runtime_error("Target for CE loss must be either 1D tensor of class indices or tensor (same shape as logits) of probabilities for each class");
        }
        Tensor new_target = Tensor(0, pred.shape());
        for (size_t i = 0; i < pred.shape()[0]; ++i) {
            new_target({i, static_cast<size_t>(target({i}))}) = 1;
        }
        target = new_target;
        // TODO: CE Loss using class index targets (need tensor indexing using tensor of indices to propagate gradient)
        // Tensor m = max(pred);
        // Tensor shifted = pred-m;
        // Tensor last_term = log_e(sum(exp(shifted), 1, true));
        // Tensor class_logits(0, target.shape(), pred.requires_grad());
        // for (size_t i = 0; i < pred.shape()[0]; ++i) {
        //     class_logits({i}) = shifted({i, static_cast<size_t>(target({i}))});
        // }
        // Tensor t3 = last_term - class_logits;
        // Tensor loss = mean(t3);
        // return loss;
    }
    if (pred.shape().size() == 1) {
        pred = unsqueeze(pred.copy(), 0);
        target = unsqueeze(target.copy(), 0);
    }
    Tensor m = max(pred);
    Tensor shifted = pred-m;
    Tensor last_term = log_e(sum(exp(shifted), 1, true));
    Tensor t3 = -sum(target * (shifted-last_term), 1);
    Tensor loss = mean(t3);
    return loss;
}
