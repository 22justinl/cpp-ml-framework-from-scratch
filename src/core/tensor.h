#pragma once

#include <initializer_list>
#include <string>
#include <vector>

class Tensor {
public:
    Tensor();
    Tensor(const std::initializer_list<float> init_data, const std::initializer_list<size_t> shape);
    Tensor(const std::initializer_list<float> init_data, const std::vector<size_t> shape);
    Tensor(const std::vector<float> init_data, const std::initializer_list<size_t> shape);
    Tensor(const std::vector<float> init_data, const std::vector<size_t> shape);
    Tensor(const float fill_val, const std::initializer_list<size_t> shape);
    Tensor(const float fill_val, const std::vector<size_t> shape);
    // TODO: copy constructor

    bool requires_grad = true;

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
    Tensor add(const Tensor& a, const Tensor& b) const;
    Tensor sub(const Tensor& a, const Tensor& b) const;
    Tensor mul(const Tensor& a, const Tensor& b) const;
    Tensor div(const Tensor& a, const Tensor& b) const;

    void zero_grad();

    std::string shape_string() const;

    const std::vector<float>& data_raw() const;
    const std::vector<float>& grad_raw() const;
    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& strides_raw() const;
    std::vector<float>& data_raw();
    std::vector<float>& grad_raw();
    std::vector<size_t>& shape();
    std::vector<size_t>& strides_raw();
private:
    std::vector<float> data_;
    std::vector<float> grad_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;

    std::vector<size_t> calculate_strides(std::vector<size_t> tensor_shape);
};
