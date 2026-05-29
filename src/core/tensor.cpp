#include "tensor.h"

#include <initializer_list>
#include <iterator>
#include <stdexcept>

#include "ops/math_ops.h"
#include "autograd/operator.h"
#include "utils/tensor_utils.h"

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
    impl_ = std::make_shared<TensorImpl>(std::vector<float>(n_el, fill_val), shape, calculate_strides(shape), requires_grad, grad);
}
Tensor::Tensor(const Tensor& other) {
    std::vector<float> new_data;
    std::vector<size_t> new_shape;
    std::vector<size_t> new_strides;
    std::copy(other.impl_->data.begin(), other.impl_->data.end(), std::back_insert_iterator(new_data));
    std::copy(other.impl_->shape.begin(), other.impl_->shape.end(), std::back_insert_iterator(new_data));
    std::copy(other.impl_->strides.begin(), other.impl_->strides.end(), std::back_insert_iterator(new_data));

    Tensor* new_grad = other.impl_->grad ? new Tensor(*other.impl_->grad) : nullptr;

    impl_ = std::make_shared<TensorImpl>(new_data, new_shape, new_strides, other.impl_->requires_grad, new_grad);
}
Tensor::Tensor(std::shared_ptr<TensorImpl> impl): impl_(impl) {}

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
Tensor Tensor::operator*(float f) const {
    return scalar_mul(*this, f);
}
Tensor operator*(float f, const Tensor& t) {
    return scalar_mul(f, t);
}
Tensor Tensor::operator/(float f) const {
    if (f == 0) { throw std::runtime_error("Divide by zero error"); }
    return scalar_mul(1/f, *this);
}
Tensor& Tensor::operator=(const Tensor& other) {
    impl_ = other.impl_;
    return *this;
}
Tensor& Tensor::operator+=(const Tensor& other) {
    *this = add(*this, other);
    return *this;
}
Tensor& Tensor::operator-=(const Tensor& other) {
    *this = sub(*this, other);
    return *this;
}
Tensor& Tensor::operator*=(const Tensor& other) {
    *this = *this * other;
    return *this;
}
Tensor& Tensor::operator/=(const Tensor& other) {
    *this = *this/other;
    return *this;
}
Tensor& Tensor::operator*=(float f) {
    *this = *this * f;
    return *this;
}
Tensor& Tensor::operator/=(float f)  {
    *this = *this / f;
    return *this;
}

Tensor Tensor::operator-() const {
    return -1.f * *this;
}

float& Tensor::at(const std::initializer_list<size_t> indices) { return at(std::vector(indices)); }
float& Tensor::at(const std::vector<size_t> indices) {
    size_t offset = calculate_offset(impl_, indices);
    return impl_->data[offset];
}
float Tensor::at(const std::initializer_list<size_t> indices) const { return at(std::vector(indices)); }
float Tensor::at(const std::vector<size_t> indices) const {
    size_t offset = calculate_offset(impl_, indices);
    return impl_->data[offset];
}

const std::vector<float>& Tensor::data_raw() const { return impl_->data; }
std::vector<float>& Tensor::data_raw() { return impl_->data; }

const Tensor& Tensor::grad() const { 
    if (!impl_->requires_grad) {
        throw std::runtime_error("Accessed grad of Tensor with requires_grad=false");
    }
    return *impl_->grad;
}
Tensor& Tensor::grad() {
    if (!impl_->requires_grad) {
        throw std::runtime_error("Accessed grad of Tensor with requires_grad=false");
    }
    return *impl_->grad;
}

const std::vector<size_t>& Tensor::shape() const { return impl_->shape; }
std::vector<size_t>& Tensor::shape() { return impl_->shape; }
std::string Tensor::shape_string() const {
    return shape_to_string(impl_->shape);
}

const std::vector<size_t>& Tensor::strides_raw() const { return impl_->strides; }
std::vector<size_t>& Tensor::strides_raw() { return impl_->strides; }

bool Tensor::requires_grad() const {
    return impl_->requires_grad;
}

std::shared_ptr<TensorImpl> Tensor::impl() const {
    return impl_;
}
void Tensor::set_impl(std::shared_ptr<TensorImpl> impl) {
    impl_ = impl;
}

Tensor Tensor::detach() const {
    // NOTE: creates copy of data, change to avoid copying later?
    return Tensor(impl_->data, impl_->shape);
}

void Tensor::set_grad_fn(std::shared_ptr<Operator> grad_fn) {
    impl_->grad_fn = grad_fn;
}

void Tensor::zero_grad() {
    if (!impl_->requires_grad) {
        throw std::runtime_error("Called zero_grad on Tensor with requires_grad=false");
    }
    std::vector<float>& grad_data = impl_->grad->impl_->data;
    for (size_t i = 0; i < grad_data.size(); ++i) {
        grad_data[i] = 0;
    }
}

void Tensor::backward() {
    backward(Tensor(1.f, impl_->shape));
}
void Tensor::backward(const Tensor& out_grad) {
    if (!impl_->requires_grad) {
        throw std::runtime_error("Called backward on Tensor with requires_grad=false");
    }
    if (!impl_->grad) {
        impl_->grad = std::make_unique<Tensor>(out_grad);
    } else {
        *impl_->grad = out_grad;
    }
    impl_->grad_fn->backward();
}

// Private functions

