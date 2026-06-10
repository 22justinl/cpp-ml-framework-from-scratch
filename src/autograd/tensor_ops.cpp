#include "tensor_ops.h"

#include "kernels/math_ops.h"
#include "kernels/tensor_ops.h"

TransposeOp::TransposeOp(const Tensor& t1, size_t dim0, size_t dim1, const Tensor& t2): dim0(dim0), dim1(dim1) {
    a = t1.impl();
    out = t2.impl();
}
Tensor TransposeOp::forward(const Tensor& t1, size_t dim0, size_t dim1) {
    return Tensor(kernels::transpose(t1.impl(), dim0, dim1));
}
void TransposeOp::backward() {
    if (a->requires_grad) {
        kernels::add_inplace(a->grad->impl(), kernels::transpose(out->grad->impl(), dim0, dim1));
    }
}
std::vector<std::shared_ptr<const TensorImpl>> TransposeOp::inputs() {
    return {a};
}
