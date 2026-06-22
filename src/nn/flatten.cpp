#include "flatten.h"

#include "ops/tensor_ops.h"

#include <vector>

namespace nn {
Flatten::Flatten(size_t start_dim, ssize_t end_dim, std::string name): start_dim(start_dim), end_dim(end_dim) {
    if (end_dim != -1 && start_dim < end_dim) {
        throw std::runtime_error("Invalid flatten start and end dimensions");
    }
    set_name(name);
}
Tensor Flatten::forward(Tensor x) {
    std::vector<size_t> new_shape(x.shape().size() - ((end_dim == -1 ? x.shape().size() : end_dim) - start_dim) + 1);
    size_t flat_dim_size = 1;
    for (size_t i = 0; i < x.shape().size(); ++i) {
        if (i < start_dim) {
            new_shape[i] = x.shape()[i];
        } else {
            flat_dim_size *= x.shape()[i];
        }
    }
    new_shape[start_dim] = flat_dim_size;
    return reshape(x, new_shape);
}
}
