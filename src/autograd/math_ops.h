#pragma once

#include "core/tensor.h"
#include "autograd/operator.h"
#include <memory>

// Element-wise addition
class AddOp: public Operator {
public:
    AddOp(const Tensor& t1, const Tensor& t2, const Tensor& t3);
    static Tensor forward(const Tensor& a, const Tensor& b);

    void backward() override;
private:
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> b;
    std::shared_ptr<const TensorImpl> out;
};
// Element-wise subtraction
class SubOp: public Operator {
public:
    SubOp(const Tensor& t1, const Tensor& t2, const Tensor& t3);
    static Tensor forward(const Tensor& a, const Tensor& b);

    void backward() override;
private:
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> b;
    std::shared_ptr<const TensorImpl> out;
};
// Element-wise multiplication
class MulOp: public Operator {
public:
    MulOp(const Tensor& t1, const Tensor& t2, const Tensor& t3);
    static Tensor forward(const Tensor& a, const Tensor& b);

    void backward() override;
private:
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> b;
    std::shared_ptr<const TensorImpl> out;
};
// Element-wise division
class DivOp: public Operator {
public:
    DivOp(const Tensor& t1, const Tensor& t2, const Tensor& t3);
    static Tensor forward(const Tensor& a, const Tensor& b);

    void backward() override;
private:
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> b;
    std::shared_ptr<const TensorImpl> out;
};

// // Tensor multiplication: (m, n)(n, l)
// Tensor matmul(const Tensor& a, const Tensor& b);
//
// // Tensor multiplication (m, n)(n) or (m, n)(n, 1)
// Tensor matvec(const Tensor& a, const Tensor& b);
//
// // (n)(n) or (n,1)(n,1) or (1,n)(1,n)
// Tensor dot(const Tensor& a, const Tensor& b);
//
// Tensor transpose(const Tensor& t);
