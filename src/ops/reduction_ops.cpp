#include "ops/reduction_ops.h"

#include "autograd/reduction_ops.h"
#include "utils/op_utils.h"

Tensor sum(const Tensor& t1, size_t dim, bool keepdim) {
    return redop_helper<SumOp>(t1, dim, keepdim);
}
Tensor mean(const Tensor& t1, size_t dim, bool keepdim) {
    return redop_helper<MeanOp>(t1, dim, keepdim);
}
Tensor max(const Tensor& t1, size_t dim, bool keepdim) {
    std::shared_ptr<TensorImpl> max_pos;
    Tensor t2 = MaxOp::forward(t1, dim, keepdim, &max_pos);
    if (t1.requires_grad()) {
        std::shared_ptr<MaxOp> op = std::make_shared<MaxOp>(t1, dim, keepdim, max_pos, t2);
        t2.set_grad_fn(op);
    }
    return t2;
}
Tensor min(const Tensor& t1, size_t dim, bool keepdim) {
    std::shared_ptr<TensorImpl> min_pos;
    Tensor t2 = MinOp::forward(t1, dim, keepdim, &min_pos);
    if (t1.requires_grad()) {
        std::shared_ptr<MinOp> op = std::make_shared<MinOp>(t1, dim, keepdim, min_pos, t2);
        t2.set_grad_fn(op);
    }
    return t2;
}
