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
