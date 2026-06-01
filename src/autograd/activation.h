#pragma once

#include "core/tensor.h"
#include "autograd/operator.h"

class SigmoidOp: public Operator {
public:
    SigmoidOp(const Tensor& t1, const Tensor& t2);
    static Tensor forward(const Tensor& t1);

    void backward() override;
private:
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> out;
};

class ReLUOp: public Operator {
public:
    ReLUOp(const Tensor& t1, const Tensor& t2);
    static Tensor forward(const Tensor& t1);

    void backward() override;
private:
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> out;
};

class SoftmaxOp: public Operator {
public:
    SoftmaxOp(const Tensor& t1, const Tensor& t2);
    static Tensor forward(const Tensor& t1);

    void backward() override;
private:
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> out;
};
