#include "tensor.h"

#include <initializer_list>
#include <stdexcept>

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

Tensor::Tensor(const std::initializer_list<float> init_data, const std::vector<size_t> shape): data_(init_data), shape_(shape) {
    if (shape_.size() > 2) {
        throw std::runtime_error("Tensors with more than 2 dimensions not implemented");
    }
    grad_ = std::vector<float>(data_.size(), 0);
    strides_ = calculate_strides(shape_);
}

Tensor::Tensor(const float fill_val, const std::initializer_list<size_t> shape): shape_(shape) {
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
float Tensor::at(const std::initializer_list<size_t> indices) const {
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
    if (!check_shape_match(a, b)) {
        throw std::runtime_error("Tensor dimension mismatch: " + a.shape_string() + " and " + b.shape_string());
    }
    Tensor res = Tensor(0, a.shape_);
    for (size_t i = 0; i < a.data_.size(); ++i) {
        res.data_[i] = a.data_[i] + b.data_[i];
    }
    return res;
}
Tensor Tensor::sub(const Tensor& a, const Tensor& b) const {
    if (!check_shape_match(a, b)) {
        throw std::runtime_error("Tensor dimension mismatch: " + a.shape_string() + " and " + b.shape_string());
    }
    Tensor res = Tensor(0, a.shape_);
    for (size_t i = 0; i < a.data_.size(); ++i) {
        res.data_[i] = a.data_[i] - b.data_[i];
    }
    return res;
}
Tensor Tensor::mul(const Tensor& a, const Tensor& b) const {
    if (!check_shape_match(a, b)) {
        throw std::runtime_error("Tensor dimension mismatch: " + a.shape_string() + " and " + b.shape_string());
    }
    Tensor res = Tensor(0, a.shape_);
    for (size_t i = 0; i < a.data_.size(); ++i) {
        res.data_[i] = a.data_[i] * b.data_[i];
    }
    return res;
}
Tensor Tensor::div(const Tensor& a, const Tensor& b) const {
    if (!check_shape_match(a, b)) {
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

Tensor Tensor::matmul(const Tensor& a, const Tensor& b) {
    // TODO: nD matmul
    if (a.shape_.size() != 2 || b.shape_.size() != 2) {
        throw std::runtime_error("Matrix multiplication only supported for 2D tensors");
    }
    if (a.shape_[a.shape_.size()-1] != b.shape_[0]) {
        throw std::runtime_error("Cannot matrix multiply tensors with shapes " + a.shape_string() + " and " + b.shape_string());
    }
    Tensor res(0.f, {a.shape_[0], b.shape_[1]});
    for (size_t i = 0; i < a.shape_[0]; ++i) {
        for (size_t j = 0; j < b.shape_[1]; ++j) {
            for (size_t k = 0; k < a.shape_[1]; ++k) {
                res({i, j}) += a({i, k}) * b({k, j});
            }
        }
    }
    return res;
}
Tensor Tensor::matvec(const Tensor& a, const Tensor& b) {
    // Expect 2D Tensor and 1D Tensor or 2D column Tensor
    if (a.shape_.size() != 2 || (b.shape_.size() != 1 && !(b.shape_.size() == 2 && b.shape_[1] == 1))) {
        throw std::runtime_error("Matrix vector multiplication expects 2D Tensor with 1D Tensor or 2D column Tensor, instead got " + a.shape_string() + " and " + b.shape_string());
    }
    if (a.shape_[1] != b.data_.size()) {
        throw std::runtime_error("Cannot matrix multiply tensors with shapes " + a.shape_string() + " and " + b.shape_string());
    }
    Tensor res(0.f, {a.shape_[0], 1});
    for (size_t i = 0; i < a.shape_[0]; ++i) {
        for (size_t j = 0; j < a.shape_[1]; ++j) {
            res.data_[i] += a({i, j})*b.data_[j];
        }
    }
    return res;
}

Tensor Tensor::dot(const Tensor& a, const Tensor& b) {
    float res = 0.f;
    if (a.shape_.size() == 0 && b.shape_.size() == 0) {
        return Tensor({res}, {1});
    }
    // a and b 1D tensors
    if (a.shape_.size() == 1 && b.shape_.size() == 1 && a.shape_[0] == b.shape_[0]) {
        for (size_t i = 0; i < a.shape_[0]; ++i) {
            res += a.data_[i] * b.data_[i];
        }
        return Tensor({res}, {1});
    }
    // a and b 2D tensors
    if (    (a.shape_.size() == 2 && b.shape_.size() == 2) &&
            ((a.shape_[0] == 1 && b.shape_[0] == 1 && a.shape_[1] == b.shape_[1]) ||
             (a.shape_[1] == 1 && b.shape_[1] == 1 && a.shape_[0] == b.shape_[0]))
        ) {
        for (size_t i = 0; i < a.data_.size(); ++i) {
            res += a.data_[i] * b.data_[i];
        }
        return Tensor({res}, {1});
    }
    throw std::runtime_error("Cannot dot product tensors with shapes " + a.shape_string() + " and " + b.shape_string());
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

bool Tensor::check_shape_match(Tensor t1, Tensor t2) const {
    if (t1.shape_.size() != t2.shape_.size()) {
        return false;
    }
    for (size_t i = 0; i < t1.shape_.size(); ++i) {
        if (t1.shape_[i] != t2.shape_[i]) {
            return false;
        }
    }

    return true;
}

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
