#include "activation.h"

#include "autograd/activation.h"
#include "utils/op_utils.h"

Tensor sigmoid(const Tensor& t1) {
    return monop_helper<class SigmoidOp>(t1);
}
Tensor relu(const Tensor& t1) {
    return monop_helper<class ReLUOp>(t1);
}
Tensor softmax(const Tensor& t1) {
    return monop_helper<class SoftmaxOp>(t1);
}
