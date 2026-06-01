#pragma once
#include "operator.h"
#include "core/tensor.h"

class SumOp: public Operator {
public:
    SumOp(const Tensor& t1, size_t dim, const Tensor& t2);
    static Tensor forward(const Tensor& t1, size_t dim);

    void backward() override;
private:
    size_t dim;
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> out;
};

class MeanOp: public Operator {
public:
    MeanOp(const Tensor& t1, size_t dim, const Tensor& t2);
    static Tensor forward(const Tensor& t1, size_t dim);

    void backward() override;
private:
    size_t dim;
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> out;
};

class MaxOp: public Operator {
public:
    MaxOp(const Tensor& t1, size_t dim, std::shared_ptr<std::vector<size_t>> indices_ptr, const Tensor& t2);
    static Tensor forward(const Tensor& t1, size_t dim, std::shared_ptr<std::vector<size_t>>* indices_pptr = nullptr);

    void backward() override;
private:
    size_t dim;
    std::shared_ptr<std::vector<size_t>> indices_ptr;
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> out;
};

class MinOp: public Operator {
public:
    MinOp(const Tensor& t1, size_t dim, std::shared_ptr<std::vector<size_t>> indices_ptr, const Tensor& t2);
    static Tensor forward(const Tensor& t1, size_t dim, std::shared_ptr<std::vector<size_t>>* indices_pptr = nullptr);

    void backward() override;
private:
    size_t dim;
    std::shared_ptr<std::vector<size_t>> indices_ptr;
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> out;
};
