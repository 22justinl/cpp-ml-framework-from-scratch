#include "utils/tensor_utils.h"

#include "core/tensor.h"

bool check_tensor_equal(Tensor t1, Tensor t2) {
    if (!check_tensor_shape_match(t1, t2)) {
        return false;
    }
    for (size_t i = 0; i < t1.shape()[0]; ++i) {
        for (size_t j = 0; j < t1.shape()[1]; ++j) {
            if (t1({i, j}) != t2({i, j})) {
                return false;
            }
        }
    }

    return true;
}
bool check_tensor_shape_match(Tensor t1, Tensor t2) {
    std::vector<size_t> t1_shape = t1.shape();
    std::vector<size_t> t2_shape = t2.shape();
    if (t1_shape.size() != t2_shape.size()) {
        return false;
    }
    for (size_t i = 0; i < t1_shape.size(); ++i) {
        if (t1_shape[i] != t2_shape[i]) {
            return false;
        }
    }

    return true;
}
