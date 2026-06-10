#include "tensor_ops.h"

#include "autograd/tensor_ops.h"

Tensor transpose(const Tensor& t1, size_t dim0, size_t dim1) {
    Tensor t2 = TransposeOp::forward(t1, dim0, dim1);
    if (t1.requires_grad() || t2.requires_grad()) {
        std::shared_ptr<TransposeOp> op = std::make_shared<TransposeOp>(t1, dim0, dim1, t2);
        t2.set_grad_fn(op);
    }
    return t2;
}
