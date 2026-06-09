#pragma once
#include "operator.h"
#include "core/tensor.h"

class SumOp: public Operator {
public:
    SumOp(const Tensor& t1, size_t dim, bool keepdim, const Tensor& t2);
    static Tensor forward(const Tensor& t1, size_t dim, bool keepdim);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    size_t dim;
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> out;
};

class MeanOp: public Operator {
public:
    MeanOp(const Tensor& t1, size_t dim, bool keepdim, const Tensor& t2);
    static Tensor forward(const Tensor& t1, size_t dim, bool keepdim);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    size_t dim;
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> out;
};

class MaxOp: public Operator {
public:
    MaxOp(const Tensor& t1, size_t dim, bool keepdim, std::shared_ptr<TensorImpl> max_pos, const Tensor& t2);
    static Tensor forward(const Tensor& t1, size_t dim, bool keepdim, std::shared_ptr<TensorImpl>* max_pos_ptr = nullptr);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    size_t dim;
    std::shared_ptr<TensorImpl> max_pos;
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> out;
};

class MinOp: public Operator {
public:
    MinOp(const Tensor& t1, size_t dim, bool keepdim, std::shared_ptr<TensorImpl> min_pos, const Tensor& t2);
    static Tensor forward(const Tensor& t1, size_t dim, bool keepdim, std::shared_ptr<TensorImpl>* min_pos_ptr = nullptr);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    size_t dim;
    std::shared_ptr<TensorImpl> min_pos;
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> out;
};
