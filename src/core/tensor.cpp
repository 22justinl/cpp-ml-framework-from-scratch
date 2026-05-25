#include "tensor.h"

#include <initializer_list>
#include <stdexcept>

#include "utils/tensor_utils.h"

Tensor::Tensor() {
    data_ = std::vector<float>(0, 0);
    grad_ = std::vector<float>(0, 0);
    shape_ = std::vector<size_t>(0, 0);
    strides_ = std::vector<size_t>(0, 0);
}

Tensor::Tensor(const std::initializer_list<float> init_data, const std::initializer_list<size_t> shape): Tensor(std::vector(init_data), std::vector(shape)) {}
Tensor::Tensor(const std::initializer_list<float> init_data, const std::vector<size_t> shape): data_(std::vector(init_data)), shape_(shape) {}
Tensor::Tensor(const std::vector<float> init_data, const std::initializer_list<size_t> shape): data_(init_data), shape_(std::vector(shape)) {}
Tensor::Tensor(const std::vector<float> init_data, const std::vector<size_t> shape): data_(init_data), shape_(shape) {
    if (shape_.size() > 2) {
        throw std::runtime_error("Tensors with more than 2 dimensions not implemented");
    }
    grad_ = std::vector<float>(data_.size(), 0);
    strides_ = calculate_strides(shape_);
}

Tensor::Tensor(const float fill_val, const std::initializer_list<size_t> shape): Tensor(fill_val, std::vector(shape)) {}
Tensor::Tensor(const float fill_val, const std::vector<size_t> shape): shape_(shape) {
    if (shape_.size() > 2) {
        throw std::runtime_error("Tensors with more than 2 dimensions not implemented");
    }
    size_t n_el = 0;
    if (shape_.size() > 0) {
        n_el = shape_[0];
        for (size_t i = 1; i < shape_.size(); ++i) {
            n_el *= shape_[i];
        }
    }
    data_ = std::vector<float>(n_el, fill_val);
    grad_ = std::vector<float>(data_.size(), 0);
    strides_ = calculate_strides(shape_);
}

float& Tensor::operator()(const std::initializer_list<size_t> indices) {
    return at(indices);
}
float Tensor::operator()(const std::initializer_list<size_t> indices) const {
    return at(indices);
}
Tensor Tensor::operator+(const Tensor& other) const {
    return add(*this, other);
}
Tensor Tensor::operator-(const Tensor& other) const {
    return sub(*this, other);
}
Tensor Tensor::operator*(const Tensor& other) const {
    return mul(*this, other);
}
Tensor Tensor::operator/(const Tensor& other) const {
    return div(*this, other);
}

float& Tensor::at(const std::initializer_list<size_t> indices) { return at(std::vector(indices)); }
float& Tensor::at(const std::vector<size_t> indices) {
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
float Tensor::at(const std::initializer_list<size_t> indices) const { return at(std::vector(indices)); }
float Tensor::at(const std::vector<size_t> indices) const {
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
Tensor Tensor::add(const Tensor& a, const Tensor& b) const {
    if (!check_tensor_shape_match(a, b)) {
        throw std::runtime_error("Tensor dimension mismatch: " + a.shape_string() + " and " + b.shape_string());
    }
    Tensor res = Tensor(0, a.shape_);
    for (size_t i = 0; i < a.data_.size(); ++i) {
        res.data_[i] = a.data_[i] + b.data_[i];
    }
    return res;
}
Tensor Tensor::sub(const Tensor& a, const Tensor& b) const {
    if (!check_tensor_shape_match(a, b)) {
        throw std::runtime_error("Tensor dimension mismatch: " + a.shape_string() + " and " + b.shape_string());
    }
    Tensor res = Tensor(0, a.shape_);
    for (size_t i = 0; i < a.data_.size(); ++i) {
        res.data_[i] = a.data_[i] - b.data_[i];
    }
    return res;
}
Tensor Tensor::mul(const Tensor& a, const Tensor& b) const {
    if (!check_tensor_shape_match(a, b)) {
        throw std::runtime_error("Tensor dimension mismatch: " + a.shape_string() + " and " + b.shape_string());
    }
    Tensor res = Tensor(0, a.shape_);
    for (size_t i = 0; i < a.data_.size(); ++i) {
        res.data_[i] = a.data_[i] * b.data_[i];
    }
    return res;
}
Tensor Tensor::div(const Tensor& a, const Tensor& b) const {
    if (!check_tensor_shape_match(a, b)) {
        throw std::runtime_error("Tensor dimension mismatch: " + a.shape_string() + " and " + b.shape_string());
    }
    Tensor res = Tensor(0, a.shape_);
    for (size_t i = 0; i < a.data_.size(); ++i) {
        if (!b.data_[i]) {
            throw std::runtime_error("Divide by zero error");
        }
        res.data_[i] = a.data_[i] / b.data_[i];
    }
    return res;
}

const std::vector<float>& Tensor::data_raw() const { return data_; }
const std::vector<float>& Tensor::grad_raw() const { return grad_; }
const std::vector<size_t>& Tensor::shape() const { return shape_; }
const std::vector<size_t>& Tensor::strides_raw() const { return strides_; }
std::vector<float>& Tensor::data_raw() { return data_; }
std::vector<float>& Tensor::grad_raw() { return grad_; }
std::vector<size_t>& Tensor::shape() { return shape_; }
std::vector<size_t>& Tensor::strides_raw() { return strides_; }

void Tensor::zero_grad() {
    for (size_t i = 0; i < data_.size(); ++i) {
        grad_[i] = 0;
    }
}

std::string Tensor::shape_string() const {
    if (shape_.size() == 0) {
        return "()";
    }
    std::string s = "(";
    for (size_t i = 0; i < shape_.size()-1; ++i) {
        s += std::to_string(shape_[i]) + ", ";
    }
    s += std::to_string(shape_[shape_.size()-1]) + ")";
    return s;
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
