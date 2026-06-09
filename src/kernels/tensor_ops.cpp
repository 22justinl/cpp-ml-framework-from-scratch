#include "kernels/tensor_ops.h"

#include "utils/tensor_utils.h"

using std::shared_ptr;
using std::make_shared;

namespace kernels {

std::shared_ptr<TensorImpl> transpose(std::shared_ptr<const TensorImpl> a) {
    if (a->shape.size() > 2) {
        throw std::runtime_error("Transpose supported for tensors up to 2 dimensions");
    }
    if (a->shape.size() == 0) {
        return make_shared<TensorImpl>(std::vector<float>(), std::vector<size_t>(), a->requires_grad);
    }
    if (a->shape.size() == 1) {
        return make_shared<TensorImpl>(a->storage->data, std::vector<size_t>({1, a->shape[0]}), std::vector<size_t>({a->shape[0], 1}), a->requires_grad);
    }
    std::vector<size_t> new_shape({a->shape[1], a->shape[0]});
    std::shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, new_shape, a->requires_grad);
    for (size_t i = 0; i < a->shape[0]; ++i) {
        for (size_t j = 0; j < a->shape[1]; ++j) {
            res->storage->data[idx_to_offset({j, i}, res->strides, res->offset)] = a->storage->data[idx_to_offset({i, j}, a->strides, a->offset)];
        }
    }
    return res;
}
}
