#include "linear.h"
#include "ops/math_ops.h"
#include "utils/nn_utils.h"

namespace nn {
Linear::Linear(size_t in_features, size_t out_features, bool bias, std::string name) {
    weight_.tensor = Tensor(0.f, {out_features, in_features}, true);
    initialize_tensor_normal(weight_.tensor, 0.f, std::sqrt(1.f/in_features));
    register_parameter(&weight_);

    if (bias) {
        bias_.tensor = Tensor(0.f, {out_features}, true);
        register_parameter(&bias_);
    }

    set_name(name);
}

Tensor Linear::forward(Tensor x) {
    return matmul(x, transpose(weight_.tensor))+bias_.tensor;
}
}
