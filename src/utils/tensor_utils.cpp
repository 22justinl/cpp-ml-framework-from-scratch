#include "utils/tensor_utils.h"

#include <cmath>
#include <iostream>
#include <vector>

void increment_idx(const Tensor& t, std::vector<size_t>& idx) {
    for (size_t dim = t.shape().size()-1; dim != SIZE_T_MAX; --dim) {
        if (idx[dim]+1 < t.shape()[dim]) {
            idx[dim] += 1;
            return;
        }
        idx[dim] = 0;
    }
}

bool check_tensor_equal(const Tensor& t1, const Tensor& t2, float eps) {
    if (!check_shape_match(t1.shape(), t2.shape())) {
        return false;
    }
    std::vector<size_t> idx(t1.shape().size(), 0);
    for (size_t i = 0; i < t1.impl()->n_el; ++i) {
        if (std::fabs(t1(idx) - t2(idx)) >= eps) {
            return false;
        }
        increment_idx(t1, idx);
    }
    return true;
}

bool check_shape_match(const std::vector<size_t> shape1, const std::vector<size_t> shape2) {
    if (shape1.size() != shape2.size()) {
        return false;
    }
    for (size_t i = 0; i < shape1.size(); ++i) {
        if (shape1[i] != shape2[i]) {
            return false;
        }
    }
    return true;
}

std::string shape_to_string(const std::vector<size_t>& shape) {
    if (shape.size() == 0) {
        return "()";
    }
    std::string s = "(";
    for (size_t i = 0; i < shape.size()-1; ++i) {
        s += std::to_string(shape[i]) + ", ";
    }
    s += std::to_string(shape[shape.size()-1]) + ")";
    return s;
}
void print_tensor(const Tensor& t1) {
    std::vector<size_t> idx(t1.shape().size(), 0);
    print_recursive(t1, idx, 0);
}
void print_recursive(const Tensor& t1, const std::vector<size_t>& idx, size_t dim) {
    if (dim == t1.shape().size()) {
        if (idx[dim-1] + 1 < t1.shape()[dim-1]) {
            std::cout << t1(idx) << ", ";
        } else {
            std::cout << t1(idx);
        }
        return;
    }
    if (dim+1 != t1.shape().size()) {
        std::cout << "[\n";
    } else {
        std::cout << "[";
    }
    for (size_t i = 0; i < t1.shape()[dim]; ++i) {
        std::vector<size_t> next_idx = idx;
        next_idx[dim] = i;
        print_recursive(t1, next_idx, dim+1);
    }
    std::cout << "]\n";
}

std::vector<size_t> calculate_strides(const std::vector<size_t>& tensor_shape) {
    std::vector strides = std::vector<size_t>(tensor_shape.size());
    if (!tensor_shape.size()) {
        return strides;
    }

    strides[tensor_shape.size()-1] = 1;
    for (size_t i = tensor_shape.size()-2; i != SIZE_T_MAX; --i) {
        strides[i] = strides[i+1] * tensor_shape[i+1];
    }
    return strides;
}
size_t calculate_n_el(const std::vector<size_t>& shape) {
    if (shape.size() == 0) {
        return 0;
    }
    size_t n_el = 0;
    n_el = shape[0];
    for (size_t i = 1; i < shape.size(); ++i) {
        n_el *= shape[i];
        if (n_el == 0) {
            return 0;
        }
    }
    return n_el;
}

// NOTE: remove later: doesn't work with views due to stride ordering
void offset_to_idx(size_t offset, const std::vector<size_t>& strides, std::vector<size_t>& idx) {
    for (size_t i = 0; i < strides.size(); ++i) {
        idx[i] = offset/strides[i];
        offset = offset % strides[i];
    }
}
// NOTE: remove later: doesn't work with views due to stride ordering
std::vector<size_t> offset_to_idx(size_t offset, const std::vector<size_t>& strides) {
    std::vector<size_t> idx;
    for (size_t i = 0; i < strides.size(); ++i) {
        idx[i] = offset/strides[i];
        offset = offset % strides[i];
    }
    return idx;
}
size_t idx_to_offset(const std::vector<size_t>& idx, const std::vector<size_t>& strides, size_t t_offset) {
    size_t offset = t_offset;
    for (size_t i = 0; i < strides.size(); ++i) {
        offset += idx[i] * strides[i];
    }
    return offset;
}
size_t idx_to_offset_checked(const std::vector<size_t>& idx, const std::vector<size_t>& shape, const std::vector<size_t>& strides, size_t t_offset) {
    if (idx.size() != shape.size()) {
        throw std::runtime_error("Index shape mismatch");
    }
    size_t offset = t_offset;
    for (size_t i = 0; i < strides.size(); ++i) {
        if (idx[i] >= shape[i]) {
            throw std::runtime_error("Index out of range");
        }
        offset += idx[i] * strides[i];
    }
    return offset;
}
