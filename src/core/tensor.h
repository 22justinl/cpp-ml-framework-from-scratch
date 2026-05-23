#pragma once

#include <initializer_list>
#include <vector>

class Tensor {
public:
    Tensor();
    Tensor(const std::initializer_list<float> init_data, const std::initializer_list<size_t> shape);

    bool requires_grad = true;

    float& operator()(const std::initializer_list<size_t> indices);
    Tensor operator+(const Tensor& other);
    Tensor operator-(const Tensor& other);
    Tensor operator*(const Tensor& other);
    Tensor operator/(const Tensor& other);

    float& at(const std::initializer_list<size_t> indices);
    Tensor add(const Tensor& a, const Tensor& b);
    Tensor sub(const Tensor& a, const Tensor& b);
    Tensor mul(const Tensor& a, const Tensor& b);
    Tensor div(const Tensor& a, const Tensor& b);

    Tensor matmul(const Tensor& a, const Tensor& b);

    const std::vector<float> data_raw() const;
    const std::vector<float> grad_raw() const;
    const std::vector<size_t> shape() const;
    const std::vector<size_t> strides_raw() const;

    void zero_grad();
private:
    std::vector<float> data_;
    std::vector<float> grad_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;

    std::vector<size_t> calculate_strides(std::vector<size_t> tensor_shape);
};
