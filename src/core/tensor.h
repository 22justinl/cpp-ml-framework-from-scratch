#pragma once

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>

struct TensorImpl;
struct TensorData;

class Operator;

class Tensor {
public:
    Tensor();
    Tensor(const std::initializer_list<float> init_data, const std::initializer_list<size_t> shape, bool requires_grad=false);
    Tensor(const std::initializer_list<float> init_data, const std::vector<size_t> shape, bool requires_grad=false);
    Tensor(const std::vector<float> init_data, const std::initializer_list<size_t> shape, bool requires_grad=false);
    Tensor(const std::vector<float> init_data, const std::vector<size_t> shape, bool requires_grad=false);
    Tensor(const float fill_val, const std::initializer_list<size_t> shape, bool requires_grad=false);
    Tensor(const float fill_val, const std::vector<size_t> shape, bool requires_grad=false);
    Tensor(const Tensor& other);
    Tensor(std::shared_ptr<TensorImpl> impl);

    float& operator()(const std::initializer_list<size_t> indices);
    float& operator()(const std::vector<size_t> indices);
    float operator()(const std::initializer_list<size_t> indices) const;
    float operator()(const std::vector<size_t> indices) const;
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator*(float f) const;
    Tensor operator/(float f) const;

    Tensor& operator=(const Tensor& other);
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);
    Tensor& operator*=(float f);
    Tensor& operator/=(float f);

    Tensor operator-() const;

    float& at(const std::initializer_list<size_t> indices);
    float& at(const std::vector<size_t> indices);
    float at(const std::initializer_list<size_t> indices) const;
    float at(const std::vector<size_t> indices) const;

    const std::vector<float>& data_raw() const;
    std::vector<float>& data_raw();

    const Tensor& grad() const;
    Tensor& grad();

    const std::vector<size_t>& shape() const;
    std::vector<size_t>& shape();
    std::string shape_string() const;

    const std::vector<size_t>& strides_raw() const;
    std::vector<size_t>& strides_raw();

    bool requires_grad() const;

    std::shared_ptr<TensorImpl> impl() const;
    void set_impl(std::shared_ptr<TensorImpl> impl);

    Tensor detach() const;

    void set_grad_fn(std::shared_ptr<Operator> grad_fn);

    void zero_grad();
    void backward();
    void backward(const Tensor& out_grad);

    void build_topo(std::shared_ptr<const TensorImpl> node, std::vector<std::shared_ptr<const TensorImpl>>& topo, std::unordered_set<std::shared_ptr<const TensorImpl>>& visited);
private:
    std::shared_ptr<TensorImpl> impl_;
};

struct TensorImpl {
    // New tensor with new shape
    TensorImpl(std::vector<float> data, std::vector<size_t> shape, bool requires_grad);
    // New tensor from existing shape
    TensorImpl(std::vector<float> data, std::vector<size_t> shape, std::vector<size_t> strides, bool requires_grad);
    // New tensor with new shape (filled)
    TensorImpl(float val, std::vector<size_t> shape, bool requires_grad);
    // New tensor from existing shape (filled)
    TensorImpl(float val, std::vector<size_t> shape, std::vector<size_t> strides, bool requires_grad);
    // View
    TensorImpl(std::shared_ptr<TensorData> storage, std::vector<size_t> shape, std::vector<size_t> strides, bool requires_grad);

    std::shared_ptr<TensorData> storage;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    size_t offset = 0;
    size_t n_el = 0;
    bool requires_grad = false;
    std::unique_ptr<Tensor> grad;
    std::shared_ptr<Operator> grad_fn = nullptr;
};

struct TensorData {
    TensorData(std::vector<float> data);
    std::vector<float> data;
};

Tensor operator*(float f, const Tensor& t);
