#include "tensor_ops.h"

#include "autograd/tensor_ops.h"

Tensor transpose(const Tensor& t1, size_t dim0, size_t dim1) {
    Tensor t2 = TransposeOp::forward(t1, dim0, dim1);
    if (t1.requires_grad()) {
        std::shared_ptr<TransposeOp> op = std::make_shared<TransposeOp>(t1, dim0, dim1, t2);
        t2.set_grad_fn(op);
    }
    return t2;
}
Tensor squeeze(const Tensor& t1, size_t dim) {
    Tensor t2 = SqueezeOp::forward(t1, dim);
    if (t1.requires_grad()) {
        std::shared_ptr<SqueezeOp> op = std::make_shared<SqueezeOp>(t1, dim, t2);
        t2.set_grad_fn(op);
    }
    return t2;
}
Tensor unsqueeze(const Tensor& t1, size_t dim) {
    Tensor t2 = UnsqueezeOp::forward(t1, dim);
    if (t1.requires_grad()) {
        std::shared_ptr<UnsqueezeOp> op = std::make_shared<UnsqueezeOp>(t1, dim, t2);
        t2.set_grad_fn(op);
    }
    return t2;
}
Tensor reshape(const Tensor& t1, const std::vector<size_t>& new_shape) {
    Tensor t2 = ReshapeOp::forward(t1, new_shape);
    if (t1.requires_grad()) {
        std::shared_ptr<ReshapeOp> op = std::make_shared<ReshapeOp>(t1, t2);
        t2.set_grad_fn(op);
    }
    return t2;
}
Tensor slice(const Tensor& t1, const std::vector<TensorIndex>& indices) {
    Tensor t2 = SliceOp::forward(t1, indices);
    if (t1.requires_grad()) {
        std::shared_ptr<SliceOp> op = std::make_shared<SliceOp>(t1, indices, t2);
        t2.set_grad_fn(op);
    }
    return t2;
}
