#pragma once
#include "operator.h"
#include "core/tensor.h"

class SumOp: public Operator {
public:
    SumOp(const Tensor& t1, size_t dim, const Tensor& t2);
    static Tensor forward(const Tensor& t1, size_t dim);

    void backward() override;
private:
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> out;
};

class MeanOp: public Operator {
public:
    MeanOp(const Tensor& t1, size_t dim, const Tensor& t2);
    static Tensor forward(const Tensor& t1, size_t dim);

    void backward() override;
private:
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> out;
};

class MaxOp: public Operator {
public:
    MaxOp(const Tensor& t1, size_t dim, const Tensor& t2);
    static Tensor forward(const Tensor& t1, size_t dim);

    void backward() override;
private:
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> out;
};

class MinOp: public Operator {
public:
    MinOp(const Tensor& t1, size_t dim, const Tensor& t2);
    static Tensor forward(const Tensor& t1, size_t dim);

    void backward() override;
private:
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> out;
};
