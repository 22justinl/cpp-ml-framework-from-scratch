#pragma once

#include <initializer_list>
#include <string>
#include <vector>

class Tensor {
public:
    Tensor();
    Tensor(const std::initializer_list<float> init_data, const std::initializer_list<size_t> shape);
    Tensor(const std::initializer_list<float> init_data, const std::vector<size_t> shape);
    Tensor(const float fill_val, const std::initializer_list<size_t> shape);
    Tensor(const float fill_val, const std::vector<size_t> shape);
    // TODO: copy constructor

    bool requires_grad = true;

    float& operator()(const std::initializer_list<size_t> indices);
    float operator()(const std::initializer_list<size_t> indices) const;
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    float& at(const std::initializer_list<size_t> indices);
    float at(const std::initializer_list<size_t> indices) const;
    Tensor add(const Tensor& a, const Tensor& b) const;
    Tensor sub(const Tensor& a, const Tensor& b) const;
    Tensor mul(const Tensor& a, const Tensor& b) const;
    Tensor div(const Tensor& a, const Tensor& b) const;

    static Tensor matmul(const Tensor& a, const Tensor& b); // (m, n)(n, l)
    static Tensor matvec(const Tensor& a, const Tensor& b); // (m, n)(n) or (m, n)(n, 1)
    static Tensor dot(const Tensor& a, const Tensor& b);    // (n)(n) or (n,1)(n,1) or (1,n)(1,n)

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

    bool check_shape_match(Tensor t1, Tensor t2) const;
    std::vector<size_t> calculate_strides(std::vector<size_t> tensor_shape);
    std::string shape_string() const;
};
