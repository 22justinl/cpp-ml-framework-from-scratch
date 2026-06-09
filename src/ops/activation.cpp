#include "activation.h"

#include "autograd/activation.h"
#include "utils/op_utils.h"

Tensor sigmoid(const Tensor& t1) {
    return monop_helper<class SigmoidOp>(t1);
}
Tensor relu(const Tensor& t1) {
    return monop_helper<class ReLUOp>(t1);
}
Tensor softmax(const Tensor& t1, size_t dim) {
    Tensor t2 = SoftmaxOp::forward(t1, dim);
    if (t1.requires_grad() || t2.requires_grad()) {
        std::shared_ptr<SoftmaxOp> op = std::make_shared<SoftmaxOp>(t1, dim, t2);
        t2.set_grad_fn(op);
    }
    return t2;
}
