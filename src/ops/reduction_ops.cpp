#include "ops/reduction_ops.h"

#include "autograd/reduction_ops.h"
#include "utils/op_utils.h"

Tensor sum(const Tensor& t1, size_t dim) {
    return redop_helper<SumOp>(t1, dim);
}
Tensor mean(const Tensor& t1, size_t dim) {
    return redop_helper<MeanOp>(t1, dim);
}
Tensor max(const Tensor& t1, size_t dim) {
    std::shared_ptr<std::vector<size_t>> indices_ptr;
    Tensor t2 = MaxOp::forward(t1, dim, &indices_ptr);
    if (t1.requires_grad() || t2.requires_grad()) {
        std::shared_ptr<MaxOp> op = std::make_shared<MaxOp>(t1, dim, indices_ptr, t2);
        t2.set_grad_fn(op);
    }
    return t2;
}
Tensor min(const Tensor& t1, size_t dim) {
    std::shared_ptr<std::vector<size_t>> indices_ptr;
    Tensor t2 = MinOp::forward(t1, dim, &indices_ptr);
    if (t1.requires_grad() || t2.requires_grad()) {
        std::shared_ptr<MinOp> op = std::make_shared<MinOp>(t1, dim, indices_ptr, t2);
        t2.set_grad_fn(op);
    }
    return t2;
}
