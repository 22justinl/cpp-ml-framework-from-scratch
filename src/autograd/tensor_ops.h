#pragma once

#include "core/tensor.h"
#include "autograd/operator.h"

class TransposeOp: public Operator {
public:
    TransposeOp(const Tensor& t1, size_t dim0, size_t dim1, const Tensor& t2);
    static Tensor forward(const Tensor& t1, size_t dim0, size_t dim1);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    std::shared_ptr<const TensorImpl> a;
    size_t dim0;
    size_t dim1;
    std::shared_ptr<const TensorImpl> out;
};

class SqueezeOp: public Operator {
public:
    SqueezeOp(const Tensor& t1, size_t dim, const Tensor& t2);
    static Tensor forward(const Tensor& t1, size_t dim);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    std::shared_ptr<const TensorImpl> a;
    size_t dim;
    std::shared_ptr<const TensorImpl> out;
};

class UnsqueezeOp: public Operator {
public:
    UnsqueezeOp(const Tensor& t1, size_t dim, const Tensor& t2);
    static Tensor forward(const Tensor& t1, size_t dim);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    std::shared_ptr<const TensorImpl> a;
    size_t dim;
    std::shared_ptr<const TensorImpl> out;
};

class ReshapeOp: public Operator {
public:
    ReshapeOp(const Tensor& t1, const Tensor& t2);
    static Tensor forward(const Tensor& t1, const std::vector<size_t>& new_shape);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    std::shared_ptr<const TensorImpl> a;
    std::shared_ptr<const TensorImpl> out;
};

class SliceOp: public Operator {
public:
    SliceOp(const Tensor& t1, const std::vector<TensorIndex>& indices, const Tensor& t2);
    static Tensor forward(const Tensor& t1, const std::vector<TensorIndex>& indices);

    void backward() override;
    std::vector<std::shared_ptr<const TensorImpl>> inputs() override;
private:
    std::shared_ptr<const TensorImpl> a;
    const std::vector<TensorIndex> indices;
    std::shared_ptr<const TensorImpl> out;
};
