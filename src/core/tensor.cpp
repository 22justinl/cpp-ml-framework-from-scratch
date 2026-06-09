#include "tensor.h"

#include <initializer_list>
#include <stdexcept>
#include <unordered_set>

#include "kernels/math_ops.h"
#include "ops/math_ops.h"
#include "autograd/operator.h"
#include "utils/tensor_utils.h"

// New tensor with new shape
TensorImpl::TensorImpl(std::vector<float> data, std::vector<size_t> shape, bool requires_grad): TensorImpl(data, shape, calculate_strides(shape), requires_grad) {}
// New tensor from existing shape
TensorImpl::TensorImpl(std::vector<float> data, std::vector<size_t> shape, std::vector<size_t> strides, bool requires_grad): shape(shape), strides(strides), requires_grad(requires_grad) {
    n_el = calculate_n_el(shape);
    if (n_el != data.size()) {
        throw std::runtime_error("Not enough data to initialize tensor");
    }
    storage = std::make_shared<TensorData>(data);
    if (requires_grad) {
        grad = std::make_unique<Tensor>(0.f, shape);
    }
}
// New tensor with new shape (filled)
TensorImpl::TensorImpl(float val, std::vector<size_t> shape, bool requires_grad): TensorImpl(val, shape, calculate_strides(shape), requires_grad) {}
// New tensor from existing shape (filled)
TensorImpl::TensorImpl(float val, std::vector<size_t> shape, std::vector<size_t> strides, bool requires_grad): shape(shape), strides(strides), requires_grad(requires_grad) {
    n_el = calculate_n_el(shape);
    storage = std::make_shared<TensorData>(std::vector<float>(n_el, val));
    if (requires_grad) {
        grad = std::make_unique<Tensor>(0.f, shape);
    }
}

// View
TensorImpl::TensorImpl(std::shared_ptr<TensorData> storage, std::vector<size_t> shape, std::vector<size_t> strides, bool requires_grad): storage(storage), shape(shape), strides(strides), requires_grad(requires_grad) {
    n_el = calculate_n_el(shape);
    if (requires_grad) {
        grad = std::make_unique<Tensor>(0.f, shape);
    }
}

TensorData::TensorData(std::vector<float> data): data(data) {}

Tensor::Tensor() {
    impl_ = std::make_shared<TensorImpl>(std::vector<float>(0, 0), std::vector<size_t>(0, 0), false);
}

Tensor::Tensor(const std::initializer_list<float> init_data, const std::initializer_list<size_t> shape, bool requires_grad): Tensor(std::vector(init_data), std::vector(shape), requires_grad) {}
Tensor::Tensor(const std::initializer_list<float> init_data, const std::vector<size_t> shape, bool requires_grad): Tensor(std::vector(init_data), shape, requires_grad) {}
Tensor::Tensor(const std::vector<float> init_data, const std::initializer_list<size_t> shape, bool requires_grad): Tensor(init_data, std::vector(shape), requires_grad) {}
Tensor::Tensor(const std::vector<float> init_data, const std::vector<size_t> shape, bool requires_grad) {
    impl_ = std::make_shared<TensorImpl>(init_data, shape, requires_grad);
}

Tensor::Tensor(const float fill_val, const std::initializer_list<size_t> shape, bool requires_grad): Tensor(fill_val, std::vector(shape), requires_grad) {}
Tensor::Tensor(const float fill_val, const std::vector<size_t> shape, bool requires_grad) {
    impl_ = std::make_shared<TensorImpl>(fill_val, shape, requires_grad);
}

Tensor::Tensor(const Tensor& other) {
    impl_ = other.impl();
}
Tensor::Tensor(std::shared_ptr<TensorImpl> impl): impl_(impl) {}

float& Tensor::operator()(const std::initializer_list<size_t> indices) {
    return at(indices);
}
float& Tensor::operator()(const std::vector<size_t>& indices) {
    return at(indices);
}
float Tensor::operator()(const std::initializer_list<size_t> indices) const {
    return at(indices);
}
float Tensor::operator()(const std::vector<size_t>& indices) const {
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
    size_t offset = idx_to_offset_checked(indices, impl_->shape, impl_->strides, impl_->offset);
    return impl_->storage->data[offset];
}
float Tensor::at(const std::initializer_list<size_t> indices) const { return at(std::vector(indices)); }
float Tensor::at(const std::vector<size_t> indices) const {
    size_t offset = idx_to_offset_checked(indices, impl_->shape, impl_->strides, impl_->offset);
    return impl_->storage->data[offset];
}

const std::vector<float>& Tensor::data_raw() const { return impl_->storage->data; }
std::vector<float>& Tensor::data_raw() { return impl_->storage->data; }

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
    return Tensor(impl_->storage->data, impl_->shape);
}

void Tensor::set_grad_fn(std::shared_ptr<Operator> grad_fn) {
    impl_->grad_fn = grad_fn;
}

void Tensor::zero_grad() {
    if (!impl_->requires_grad) {
        throw std::runtime_error("Called zero_grad on Tensor with requires_grad=false");
    }
    kernels::zero_inplace(impl_->grad->impl_);
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
    std::vector<std::shared_ptr<const TensorImpl>> topo;
    std::unordered_set<std::shared_ptr<const TensorImpl>> visited;

    build_topo(impl_, topo, visited);
    for (size_t i = topo.size()-1; i != SIZE_MAX; --i) {
        if (topo[i]->grad_fn) {
            topo[i]->grad_fn->backward();
        }
    }
}

void Tensor::build_topo(std::shared_ptr<const TensorImpl> node, std::vector<std::shared_ptr<const TensorImpl>>& topo, std::unordered_set<std::shared_ptr<const TensorImpl>>& visited) {
    if (visited.contains(node)) {
        return;
    }
    visited.insert(node);
    std::shared_ptr<Operator> op = node->grad_fn;
    if (op) {
        for (std::shared_ptr<const TensorImpl> ti : op->inputs()) {
            build_topo(ti, topo, visited);
        }
    }
    topo.push_back(node);
}

// Private functions

