#include "kernels/tensor_ops.h"

#include "utils/tensor_utils.h"

using std::shared_ptr;
using std::make_shared;

namespace kernels {

shared_ptr<TensorImpl> transpose(shared_ptr<const TensorImpl> a, size_t dim0, size_t dim1) {
    if (dim0 >= a->shape.size() || dim1 >= a->shape.size()) {
        throw std::runtime_error("Transpose dim argument larger than tensor dimension");
    }
    if (a->shape.size() < 2 || dim0 == dim1) {
        return make_shared<TensorImpl>(a->storage, a->shape, a->strides, a->requires_grad);
    }
    std::vector<size_t> new_shape = a->shape;
    std::vector<size_t> new_strides = a->strides;
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);
    return make_shared<TensorImpl>(a->storage, new_shape, new_strides, a->requires_grad);
}
std::shared_ptr<TensorImpl> squeeze(std::shared_ptr<const TensorImpl> a, size_t dim) {
    if (dim >= a->shape.size()) {
        throw std::runtime_error("Squeeze dim argument larger than tensor dimension");
    }
    if (a->shape[dim] == 1) {
        std::vector<size_t> new_shape = a->shape;
        std::vector<size_t> new_strides = a->strides;
        new_shape.erase(new_shape.begin()+dim);
        new_strides.erase(new_strides.begin()+dim);
        return make_shared<TensorImpl>(a->storage, new_shape, new_strides, a->requires_grad);
    }
    return make_shared<TensorImpl>(a->storage, a->shape, a->strides, a->requires_grad);
}
std::shared_ptr<TensorImpl> unsqueeze(std::shared_ptr<const TensorImpl> a, size_t dim) {
    if (dim > a->shape.size()) {
        throw std::runtime_error("Unsqueeze dim argument larger than tensor dimension");
    }
    std::vector<size_t> new_shape = a->shape;
    std::vector<size_t> new_strides = a->strides;
    new_shape.insert(new_shape.begin()+dim, 1);
    new_strides.insert(new_strides.begin()+dim, dim < a->shape.size() ? a->strides[dim]*a->shape[dim] : 1);
    return make_shared<TensorImpl>(a->storage, new_shape, new_strides, a->requires_grad);
}
std::shared_ptr<TensorImpl> reshape(std::shared_ptr<const TensorImpl> a, const std::vector<size_t>& new_shape) {
    if (a->n_el != calculate_n_el(new_shape)) {
        throw std::runtime_error("Cannot reshape tensor to shape with different number of elements");
    }
    if (check_shape_match(a->shape, new_shape) || a->shape.size() == 0) {
        return make_shared<TensorImpl>(a->storage, a->shape, a->strides, a->requires_grad);
    }

    // 1. find groups -> array "groups" of start and end indices of groups (start inclusive, end exclusive)
    // 2. iterate through new shape starting from the right, group dims together, calculating n_el as we go
    //      if n_el < n_el of current group in "groups": continue
    //      if n_el > n_el of current group in "groups": invalid, need to copy for reshape
    //      if n_el = n_el of current group in "groups": valid, start new group, n_el = 1
    //          (how to deal with dimensions with size 1?)

    // find "contiguous groups" (shape[dim] and shape[dim+1] in contiguous group if strides[dim]=strides[dim+1]*shape[dim+1])
    size_t group_count = 1;                                 // only need n_el of groups
    std::vector<size_t> group_n_el{a->shape[0]};
    for (size_t d = 0; d < a->shape.size(); ++d) {
        if (a->strides[d] == a->strides[d+1] * a->shape[d+1]) {
            group_n_el[group_count-1] *= a->shape[d+1];     // update group
        } else {
            ++group_count;                                  // new group found
            group_n_el.push_back(a->shape[d+1]);            // create new group
        }
    }

    // check if reshape possible without copying
    // FIX: need to calculate strides within groups
    //      can calculate as we go if we iterate starting from the right?
    bool valid_view = true;
    size_t dim = 0;
    std::vector<size_t> new_strides(new_shape.size(), 0);
    for (size_t g = 0; g < group_count && dim < new_shape.size(); ++g) {
        size_t n_el = new_shape[dim];
        while (dim < new_shape.size() && n_el * new_shape[dim+1] <= group_n_el[g]) {
            n_el *= new_shape[dim+1];
            ++dim;
        }
        if (n_el != group_n_el[g]) {
            // current group n_el incompatible with contiguous group g
            valid_view = false;
            break;
        }
    }
    if (dim < new_shape.size()) {
        // new_shape didn't fit into groups
        valid_view = false;
    }

    if (valid_view) {
        return make_shared<TensorImpl>(a->storage, new_shape, calculate_strides(new_shape), a->requires_grad);
        // return make_shared<TensorImpl>(a->storage, new_shape, new_strides, a->requires_grad);
    }
    // invalid view, copy and reshape
    // copy to "contiguous" tensor
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, a->shape, a->requires_grad);
    std::vector<size_t> idx(a->shape.size(), 0);
    for (size_t i = 0; i <  a->n_el; ++i) {
        res->storage->data[idx_to_offset(idx, res->strides, res->offset)] = a->storage->data[idx_to_offset(idx, a->strides, a->offset)];
        increment_idx(res, idx);
    }
    // reshape
    res->shape = new_shape;
    res->strides = calculate_strides(new_shape);
    if (a->grad) {
        a->grad->impl()->shape = res->shape;
        a->grad->impl()->strides = res->strides;
    }

    return res;
}
}
