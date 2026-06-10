#include "kernels/tensor_ops.h"

#include "utils/tensor_utils.h"

using std::shared_ptr;
using std::make_shared;

namespace kernels {

shared_ptr<TensorImpl> transpose(shared_ptr<const TensorImpl> a, size_t dim0, size_t dim1) {
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(a->storage, a->shape, a->strides, a->requires_grad);
    if (a->shape.size() < 2 || dim0 == dim1) {
        return res;
    }
    if (dim0 >= a->shape.size() || dim1 >= a->shape.size()) {
        throw std::runtime_error("Transpose dim argument larger than tensor dimension");
    }
    std::swap(res->shape[dim0], res->shape[dim1]);
    std::swap(res->strides[dim0], res->strides[dim1]);
    if (res->grad) {
        std::swap(res->grad->impl()->shape[dim0], res->grad->impl()->shape[dim1]);
        std::swap(res->grad->impl()->strides[dim0], res->grad->impl()->strides[dim1]);
    }
    return res;
}
shared_ptr<TensorImpl> view(std::shared_ptr<const TensorImpl> a, const std::vector<size_t>& new_shape, const std::vector<size_t>& new_strides, const std::vector<size_t>& offset_idx) {
    return make_shared<TensorImpl>(a->storage, new_shape, new_strides, idx_to_offset(offset_idx, a->strides, a->offset));
}
}
