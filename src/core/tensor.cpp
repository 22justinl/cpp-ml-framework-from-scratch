#include "tensor.h"

#include <exception>
#include <iostream>

Tensor::Tensor() {
    data_ = std::vector<float>(0, 0);
    grad_ = std::vector<float>(0, 0);
    shape_ = std::vector<size_t>(0, 0);
    strides_ = std::vector<size_t>(0, 0);
}

Tensor::Tensor(const std::initializer_list<float> init_data, const std::initializer_list<size_t> shape): data_(init_data), shape_(shape) {
    if (shape_.size() > 2) {
        throw std::runtime_error("Tensors with more than 2 dimensions not implemented");
    }

    grad_ = std::vector<float>(data_.size(), 0);
    strides_ = calculate_strides(shape_);
}

float& Tensor::operator()(const std::initializer_list<size_t> indices) {
    return at(indices);
}
Tensor Tensor::operator+(const Tensor& other) {
    // TODO:
    return Tensor({}, {});
}
Tensor Tensor::operator-(const Tensor& other) {
    // TODO:
    return Tensor({}, {});
}
Tensor Tensor::operator*(const Tensor& other) {
    // TODO:
    return Tensor({}, {});
}
Tensor Tensor::operator/(const Tensor& other) {
    // TODO:
    return Tensor({}, {});
}

float& Tensor::at(const std::initializer_list<size_t> indices) {
    if (indices.size() != shape_.size()) {
        throw std::runtime_error("Index shape mismatch");
    }
    size_t offset = 0;
    for (size_t i = 0; i < shape_.size(); ++i) {
        size_t idx = *(indices.begin()+i);
        if (idx >= shape_[i]) {
            throw std::runtime_error("Index out of range");
        }
        offset += strides_[i] * idx;
    }
    return data_[offset];
}
Tensor Tensor::add(const Tensor& a, const Tensor& b) {
    // TODO:
    return Tensor({}, {});
}
Tensor Tensor::sub(const Tensor& a, const Tensor& b) {
    // TODO:
    return Tensor({}, {});
}
Tensor Tensor::mul(const Tensor& a, const Tensor& b) {
    // TODO:
    return Tensor({}, {});
}
Tensor Tensor::div(const Tensor& a, const Tensor& b) {
    // TODO:
    return Tensor({}, {});
}

Tensor Tensor::matmul(const Tensor& a, const Tensor& b) {
    // TODO:
    return Tensor({}, {});
}

const std::vector<float> Tensor::data_raw() const { return data_; }
const std::vector<float> Tensor::grad_raw() const { return grad_; }
const std::vector<size_t> Tensor::shape() const { return shape_; }
const std::vector<size_t> Tensor::strides_raw() const { return strides_; }

void Tensor::zero_grad() {
    for (size_t i = 0; i < data_.size(); ++i) {
        grad_[i] = 0;
    }
}

// Private functions

std::vector<size_t> Tensor::calculate_strides(std::vector<size_t> tensor_shape) {
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
