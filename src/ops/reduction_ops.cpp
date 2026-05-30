#include "ops/reduction_ops.h"

#include "autograd/reduction_ops.h"
#include "utils/op_utils.h"

Tensor sum(const Tensor& t1, size_t dim) {
    return redop_helper<SumOp>(t1, dim);
}
Tensor mean(const Tensor& t1, size_t dim) {
    return redop_helper<MeanOp>(t1, dim);
}
// Tensor max(const Tensor& t1, size_t dim) {
//     return redop_helper<MaxOp>(t1, dim);
// }
//
// Tensor min(const Tensor& t1, size_t dim) {
//     return redop_helper<MinOp>(t1, dim);
// }
