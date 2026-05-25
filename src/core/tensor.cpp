#include "tensor.h"

#include <initializer_list>
#include <stdexcept>

#include "ops/math_ops.h"

Tensor::Tensor() {
    impl_ = std::make_shared<TensorImpl>(TensorImpl(std::vector<float>(0, 0), std::vector<size_t>(0, 0), std::vector<size_t>(0, 0), false, nullptr));
}

Tensor::Tensor(const std::initializer_list<float> init_data, const std::initializer_list<size_t> shape, bool requires_grad): Tensor(std::vector(init_data), std::vector(shape), requires_grad) {}
Tensor::Tensor(const std::initializer_list<float> init_data, const std::vector<size_t> shape, bool requires_grad): Tensor(std::vector(init_data), shape, requires_grad) {}
Tensor::Tensor(const std::vector<float> init_data, const std::initializer_list<size_t> shape, bool requires_grad): Tensor(init_data, std::vector(shape), requires_grad) {}
Tensor::Tensor(const std::vector<float> init_data, const std::vector<size_t> shape, bool requires_grad) {
    if (shape.size() > 2) {
        throw std::runtime_error("Tensors with more than 2 dimensions not implemented");
    }
    Tensor* grad = nullptr;
    if (requires_grad) {
        grad = new Tensor(0.f, shape);
    }
    impl_ = std::make_shared<TensorImpl>(TensorImpl(init_data, shape, calculate_strides(shape), requires_grad, grad));
}

Tensor::Tensor(const float fill_val, const std::initializer_list<size_t> shape, bool requires_grad): Tensor(fill_val, std::vector(shape), requires_grad) {}
Tensor::Tensor(const float fill_val, const std::vector<size_t> shape, bool requires_grad) {
    if (shape.size() > 2) {
        throw std::runtime_error("Tensors with more than 2 dimensions not implemented");
    }
    size_t n_el = 0;
    if (shape.size() > 0) {
        n_el = shape[0];
        for (size_t i = 1; i < shape.size(); ++i) {
            n_el *= shape[i];
        }
    }
    Tensor* grad = nullptr;
    if (requires_grad) {
        grad = new Tensor(0.f, shape);
    }
    impl_ = std::make_shared<TensorImpl>(TensorImpl(std::vector<float>(n_el, fill_val), shape, calculate_strides(shape), requires_grad, grad));
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
    if (indices.size() != impl_->shape.size()) {
        throw std::runtime_error("Index shape mismatch");
    }
    size_t offset = 0;
    for (size_t i = 0; i < impl_->shape.size(); ++i) {
        size_t idx = *(indices.begin()+i);
        if (idx >= impl_->shape[i]) {
            throw std::runtime_error("Index out of range");
        }
        offset += impl_->strides[i] * idx;
    }
    return impl_->data[offset];
}
float Tensor::at(const std::initializer_list<size_t> indices) const { return at(std::vector(indices)); }
float Tensor::at(const std::vector<size_t> indices) const {
    if (indices.size() != impl_->shape.size()) {
        throw std::runtime_error("Index shape mismatch");
    }
    size_t offset = 0;
    for (size_t i = 0; i < impl_->shape.size(); ++i) {
        size_t idx = *(indices.begin()+i);
        if (idx >= impl_->shape[i]) {
            throw std::runtime_error("Index out of range");
        }
        offset += impl_->strides[i] * idx;
    }
    return impl_->data[offset];
}

const std::vector<float>& Tensor::data_raw() const { return impl_->data; }
const Tensor& Tensor::grad() const { 
    if (!impl_->requires_grad) {
        throw std::runtime_error("Accessed grad of Tensor with requires_grad=false");
    }
    return *impl_->grad;
}
const std::vector<size_t>& Tensor::shape() const { return impl_->shape; }
const std::vector<size_t>& Tensor::strides_raw() const { return impl_->strides; }
std::vector<float>& Tensor::data_raw() { return impl_->data; }
Tensor& Tensor::grad() {
    if (!impl_->requires_grad) {
        throw std::runtime_error("Accessed grad of Tensor with requires_grad=false");
    }
    return *impl_->grad;
}
std::vector<size_t>& Tensor::shape() { return impl_->shape; }
std::vector<size_t>& Tensor::strides_raw() { return impl_->strides; }

void Tensor::zero_grad() {
    std::vector<float>& grad_data = impl_->grad->impl_->data;
    for (size_t i = 0; i < grad_data.size(); ++i) {
        grad_data[i] = 0;
    }
}

std::string Tensor::shape_string() const {
    if (impl_->shape.size() == 0) {
        return "()";
    }
    std::string s = "(";
    for (size_t i = 0; i < impl_->shape.size()-1; ++i) {
        s += std::to_string(impl_->shape[i]) + ", ";
    }
    s += std::to_string(impl_->shape[impl_->shape.size()-1]) + ")";
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
