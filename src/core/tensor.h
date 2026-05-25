#pragma once

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

struct TensorImpl;

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
    // TODO: copy constructor

    float& operator()(const std::initializer_list<size_t> indices);
    float& operator()(const std::vector<size_t> indices);
    float operator()(const std::initializer_list<size_t> indices) const;
    float operator()(const std::vector<size_t> indices) const;
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    float& at(const std::initializer_list<size_t> indices);
    float& at(const std::vector<size_t> indices);
    float at(const std::initializer_list<size_t> indices) const;
    float at(const std::vector<size_t> indices) const;

    void zero_grad();

    std::string shape_string() const;

    const std::vector<float>& data_raw() const;
    const Tensor& grad() const;
    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& strides_raw() const;
    std::vector<float>& data_raw();
    Tensor& grad();
    std::vector<size_t>& shape();
    std::vector<size_t>& strides_raw();
private:
    std::shared_ptr<TensorImpl> impl_;
    std::vector<size_t> calculate_strides(std::vector<size_t> tensor_shape);
};

struct TensorImpl {
    TensorImpl(
            std::vector<float> data, 
            std::vector<size_t> shape, 
            std::vector<size_t> strides,
            bool requires_grad,
            Tensor* grad
            ): 
        data(data),
        shape(shape),
        strides(strides),
        requires_grad(requires_grad),
        grad(grad) {}
    std::vector<float> data;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    bool requires_grad = false;
    std::unique_ptr<Tensor> grad;
    std::shared_ptr<Operator> creator = nullptr;
};
