#include "utils/tensor_utils.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

bool check_tensor_equal(const Tensor& t1, const Tensor& t2, float eps) {
    if (!check_shape_match(t1.shape(), t2.shape())) {
        return false;
    }
    std::vector<size_t> idx(t1.shape().size(), 0);
    const std::vector<size_t>& shape = t1.shape();

    size_t a_offset = t1.impl()->offset;
    size_t b_offset = t2.impl()->offset;

    const std::vector<size_t>& a_strides = t1.impl()->strides;
    const std::vector<size_t>& b_strides = t2.impl()->strides;

    const std::vector<float>& a_data = t1.impl()->storage->data;
    const std::vector<float>& b_data = t2.impl()->storage->data;

    const size_t n_el = t1.impl()->n_el;
    for (size_t i = 0; i < n_el; ++i) {
        if (std::fabs(a_data[a_offset] - b_data[b_offset]) >= eps) {
            return false;
        }
        increment_offset_unary_op(idx, shape, a_offset, a_strides, b_offset, b_strides);
    }
    return true;
}

bool check_shape_match(const std::vector<size_t> shape1, const std::vector<size_t> shape2, size_t ignore_suffix_count) {
    if (shape1.size() != shape2.size()) {
        return false;
    }
    for (size_t i = 0; i < shape1.size()-ignore_suffix_count; ++i) {
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
    std::cout << std::setprecision(10);
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
    for (size_t i = tensor_shape.size()-2; i != SIZE_MAX; --i) {
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

size_t idx_to_offset(const std::vector<size_t>& idx, const std::vector<size_t>& shape, const std::vector<size_t>& strides, size_t t_offset) {
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

void increment_offset(std::vector<size_t>& idx, const std::vector<size_t>& shape, size_t& offset, const std::vector<size_t>& strides) {
    for (size_t dim = shape.size(); dim -- > 0;) {
        if (idx[dim]+1 < shape[dim]) {
            idx[dim] += 1;
            offset += strides[dim];
            return;
        }
        offset -= strides[dim]*(shape[dim]-1);
        idx[dim] = 0;
    }
}

// increment_offset but increment offsets for 2 tensors (input and output of unary operation)
void increment_offset_unary_op(std::vector<size_t>& idx, const std::vector<size_t>& shape,
        size_t& a_offset, const std::vector<size_t>& a_strides,
        size_t& res_offset, const std::vector<size_t>& res_strides) {
    for (size_t dim = shape.size(); dim -- > 0;) {
        if (idx[dim]+1 < shape[dim]) {
            idx[dim] += 1;
            a_offset += a_strides[dim];
            res_offset += res_strides[dim];
            return;
        }
        a_offset -= a_strides[dim]*(shape[dim]-1);
        res_offset -= res_strides[dim]*(shape[dim]-1);
        idx[dim] = 0;
    }
}

// increment_offset but increment offsets for 3 tensors (inputs and output of binary operation)
void increment_offset_binary_op(std::vector<size_t>& idx, const std::vector<size_t>& shape,
        size_t& a_offset, const std::vector<size_t>& a_strides,
        size_t& b_offset, const std::vector<size_t>& b_strides,
        size_t& res_offset, const std::vector<size_t>& res_strides) {
    for (size_t dim = shape.size(); dim -- > 0;) {
        if (idx[dim]+1 < shape[dim]) {
            idx[dim] += 1;
            a_offset += a_strides[dim];
            b_offset += b_strides[dim];
            res_offset += res_strides[dim];
            return;
        }
        a_offset -= a_strides[dim]*(shape[dim]-1);
        b_offset -= b_strides[dim]*(shape[dim]-1);
        res_offset -= res_strides[dim]*(shape[dim]-1);
        idx[dim] = 0;
    }
}

void increment_offset_matmul_op(std::vector<size_t>& idx, const std::vector<size_t>& shape,
        size_t& a_offset, const std::vector<size_t>& a_strides,
        size_t& b_offset, const std::vector<size_t>& b_strides,
        size_t& c_offset, const std::vector<size_t>& c_strides) {
    for (size_t dim = shape.size()-2; dim -- > 0;) {
        if (idx[dim]+1 < shape[dim]) {
            idx[dim] += 1;
            a_offset += a_strides[dim];
            b_offset += b_strides[dim];
            c_offset += c_strides[dim];
            return;
        }
        a_offset -= a_strides[dim]*(shape[dim]-1);
        b_offset -= b_strides[dim]*(shape[dim]-1);
        c_offset -= c_strides[dim]*(shape[dim]-1);
        idx[dim] = 0;
    }
}

bool is_contiguous(std::shared_ptr<const TensorImpl> a) {
    if (a->strides[a->shape.size()-1] != 1) {
        return false;
    }
    size_t expected = 1;
    for (size_t i = a->shape.size()-1; i-->0;) {
        expected *= a->shape[i+1];
        if (a->strides[i] != expected) {
            return false;
        }
    }
    return true;
}
