#pragma once

#include "core/tensor.h"
#include "autograd/operator.h"

class SigmoidOp: public Operator {
public:
    SigmoidOp(const Tensor& t1, const Tensor& t2);
    static Tensor forward(const Tensor& t1);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> out;
};

class ReLUOp: public Operator {
public:
    ReLUOp(const Tensor& t1, const Tensor& t2);
    static Tensor forward(const Tensor& t1);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> out;
};

class SoftmaxOp: public Operator {
public:
    SoftmaxOp(const Tensor& t1, size_t dim, const Tensor& t2);
    static Tensor forward(const Tensor& t1, size_t dim);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    std::shared_ptr<const TensorImpl> a;
    size_t dim;
    std::shared_ptr<const TensorImpl> out;
};
