#include "linear.h"
#include "ops/math_ops.h"
#include "ops/tensor_ops.h"
#include "utils/nn_utils.h"

namespace nn {
Linear::Linear(size_t in_features, size_t out_features, bool bias, std::string name): bias(bias) {
    weight_.tensor = Tensor(0.f, {in_features, out_features}, true);
    initialize_tensor_normal(weight_.tensor, 0.f, std::sqrt(1.f/in_features));
    register_parameter(&weight_);

    if (bias) {
        bias_.tensor = Tensor(0.f, {out_features}, true);
        register_parameter(&bias_);
    }

    set_name(name);
}

Tensor Linear::forward(Tensor x) {
    if (!bias) {
        return matmul(x, weight_.tensor);
    }
    return mmadd(x, weight_.tensor, bias_.tensor);
}
}
