#include "loss.h"
#include "ops/loss.h"

namespace nn {
MSELoss::MSELoss(std::string name) {
    set_name(name);
}
Tensor MSELoss::forward(Tensor pred, Tensor target) {
    return mse_loss(pred, target);
}

CrossEntropyLoss::CrossEntropyLoss(std::string name) {
    set_name(name);
}
Tensor CrossEntropyLoss::forward(Tensor pred, Tensor target) {
    return cross_entropy_loss(pred, target);
}
}
