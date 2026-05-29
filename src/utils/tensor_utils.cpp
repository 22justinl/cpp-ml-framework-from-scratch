#include "utils/tensor_utils.h"

#include <iostream>
#include <vector>

bool check_tensor_equal(const Tensor& t1, const Tensor& t2) {
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
bool check_tensor_shape_match(const Tensor& t1, const Tensor& t2) {
    return check_tensorimpl_shape_match(t1.impl(), t2.impl());
}
bool check_tensorimpl_shape_match(std::shared_ptr<const TensorImpl> t1, std::shared_ptr<const TensorImpl> t2) {
    std::vector<size_t> t1_shape = t1->shape;
    std::vector<size_t> t2_shape = t2->shape;
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
        std::cout << "[";
    for (size_t i = 0; i < t1.shape()[0]; ++i) {
        std::cout << "[";
        for (size_t j = 0; j < t1.shape()[1]; ++j) {
            std::cout << t1({i, j}) << "\t\t";
        }
        std::cout << "]" << std::endl;
    }
}

std::vector<size_t> calculate_strides(std::vector<size_t> tensor_shape) {
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
size_t calculate_offset(std::shared_ptr<const TensorImpl> impl, const std::vector<size_t>& indices) {
    if (indices.size() != impl->shape.size()) {
        throw std::runtime_error("Index shape mismatch");
    }
    size_t offset = 0;
    for (size_t i = 0; i < impl->shape.size(); ++i) {
        size_t idx = *(indices.begin()+i);
        if (idx >= impl->shape[i]) {
            throw std::runtime_error("Index out of range");
        }
        offset += impl->strides[i] * idx;
    }
    return offset;
}
std::shared_ptr<TensorImpl> create_tensorimpl(const std::vector<float>& data, const std::vector<size_t>& shape, const std::vector<size_t>& strides, bool requires_grad) {
    if (requires_grad) {
        return std::make_shared<TensorImpl>(data, shape, strides, requires_grad, new Tensor(0.f, shape));
    }
    return std::make_shared<TensorImpl>(data, shape, strides, requires_grad, nullptr);
}
