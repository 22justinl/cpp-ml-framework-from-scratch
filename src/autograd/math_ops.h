#pragma once

#include "core/tensor.h"
#include "autograd/operator.h"

// Element-wise addition
class AddOp: public Operator {
public:
    AddOp(const Tensor& t1, const Tensor& t2, const Tensor& t3);
    static Tensor forward(const Tensor& t1, const Tensor& t2);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> b;
    std::shared_ptr<const TensorImpl> out;
};
// Element-wise subtraction
class SubOp: public Operator {
public:
    SubOp(const Tensor& t1, const Tensor& t2, const Tensor& t3);
    static Tensor forward(const Tensor& t1, const Tensor& t2);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> b;
    std::shared_ptr<const TensorImpl> out;
};
// Element-wise multiplication
class MulOp: public Operator {
public:
    MulOp(const Tensor& t1, const Tensor& t2, const Tensor& t3);
    static Tensor forward(const Tensor& t1, const Tensor& t2);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> b;
    std::shared_ptr<const TensorImpl> out;
};
// Element-wise division
class DivOp: public Operator {
public:
    DivOp(const Tensor& t1, const Tensor& t2, const Tensor& t3);
    static Tensor forward(const Tensor& t1, const Tensor& t2);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> b;
    std::shared_ptr<const TensorImpl> out;
};

// Tensor scalar multiplication
class ScalarMulOp: public Operator {
public:
    ScalarMulOp(float f, const Tensor& t1, const Tensor& t2);
    static Tensor forward(float f, const Tensor& t1);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    float f;
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> out;
};

// Matrix multiplication: (m, n)(n, l)
class MatMulOp: public Operator {
public:
    MatMulOp(const Tensor& t1, const Tensor& t2, const Tensor& t3);
    static Tensor forward(const Tensor& t1, const Tensor& t2);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> b;
    std::shared_ptr<const TensorImpl> out;
};
class ScaledMatMulOp: public Operator {
public:
    ScaledMatMulOp(float alpha, const Tensor& t1, const Tensor& t2, const Tensor& t3);
    static Tensor forward(float alpha, const Tensor& t1, const Tensor& t2);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    float alpha;
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> b;
    std::shared_ptr<const TensorImpl> out;
};
class MMAddOp: public Operator {
public:
    MMAddOp(const Tensor& t1, const Tensor& t2, const Tensor& t3, const Tensor& t4);
    static Tensor forward(const Tensor& t1, const Tensor& t2, const Tensor& t3);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> b;
    std::shared_ptr<const TensorImpl> c;
    std::shared_ptr<const TensorImpl> out;
};
class MMAddGeneralOp: public Operator {
public:
    MMAddGeneralOp(float alpha, const Tensor& t1, const Tensor& t2, float beta, const Tensor& t3, const Tensor& t4);
    static Tensor forward(float alpha, const Tensor& t1, const Tensor& t2, float beta, const Tensor& t3);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    float alpha;
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> b;
    float beta;
    std::shared_ptr<const TensorImpl> c;
    std::shared_ptr<const TensorImpl> out;
};

// 1D Tensor/vector dot product
class DotOp: public Operator {
public:
    DotOp(const Tensor& t1, const Tensor& t2, const Tensor& t3);
    static Tensor forward(const Tensor& t1, const Tensor& t2);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> b;
    std::shared_ptr<const TensorImpl> out;
};

class PowerOp: public Operator {
public:
    PowerOp(const Tensor& t1, float x, const Tensor& t2);
    static Tensor forward(const Tensor& t1, float x);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    std::shared_ptr<const TensorImpl> a;
    float x;
    std::shared_ptr<const TensorImpl> out;
};

class ExpOp: public Operator {
public:
    ExpOp(const Tensor& t1, const Tensor& t2);
    static Tensor forward(const Tensor& t1);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> out;
};
// Natural log
class LogOp: public Operator {
public:
    LogOp(const Tensor& t1, const Tensor& t2);
    static Tensor forward(const Tensor& t1);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> out;
};
