#include "optimizer.h"

void Optimizer::zero_grad() {
    for (nn::Parameter* p_ptr : parameters_) {
        p_ptr->tensor.zero_grad();
    }
}
