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
        return make_shared<TensorImpl>(a->storage, a->shape, a->strides, a->offset, a->requires_grad);
    }
    std::vector<size_t> new_shape = a->shape;
    std::vector<size_t> new_strides = a->strides;
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);
    return make_shared<TensorImpl>(a->storage, new_shape, new_strides, a->offset, a->requires_grad);
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
        return make_shared<TensorImpl>(a->storage, new_shape, new_strides, a->offset, a->requires_grad);
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
    return make_shared<TensorImpl>(a->storage, new_shape, new_strides, a->offset, a->requires_grad);
}
std::shared_ptr<TensorImpl> reshape(std::shared_ptr<const TensorImpl> a, const std::vector<size_t>& new_shape) {
    if (a->n_el != calculate_n_el(new_shape)) {
        throw std::runtime_error("Cannot reshape tensor to shape with different number of elements");
    }
    if (check_shape_match(a->shape, new_shape) || a->shape.size() == 0) {
        return make_shared<TensorImpl>(a->storage, a->shape, a->strides, a->offset, a->requires_grad);
    }

    // find "contiguous groups" (shape[dim] and shape[dim+1] in contiguous group if strides[dim]=strides[dim+1]*shape[dim+1])
    size_t group_count = 1;                                 // only need n_el of groups
    std::vector<size_t> group_n_el{a->shape[0]};
    std::vector<size_t> group_last_stride{};
    for (size_t d = 0; d < a->shape.size()-1; ++d) {
        if (a->strides[d] == a->strides[d+1] * a->shape[d+1]) {
            group_n_el[group_count-1] *= a->shape[d+1];     // update group
        } else {
            group_last_stride.push_back(a->strides[d]);
            ++group_count;                                  // new group found
            group_n_el.push_back(a->shape[d+1]);            // create new group
        }
    }
    group_last_stride.push_back(a->strides[a->shape.size()-1]);

    // check if reshape possible without copying, calculate strides for each group
    bool valid_view = true;
    size_t dim = new_shape.size();
    std::vector<size_t> new_strides(new_shape.size(), 0);
    for (size_t g = group_count; g-- && dim != 0;) {
        --dim;
        size_t n_el = new_shape[dim];
        new_strides[dim] = group_last_stride[g];
        while (dim != 0 && n_el * new_shape[dim-1] <= group_n_el[g]) {
            new_strides[dim-1] = new_strides[dim] * new_shape[dim];
            n_el *= new_shape[dim-1];
            --dim;
        }
        if (n_el != group_n_el[g]) {
            // current group n_el incompatible with contiguous group g
            valid_view = false;
            break;
        }
    }
    if (dim != 0) {
        // new_shape didn't fit into groups
        valid_view = false;
    }

    if (valid_view) {
        // return make_shared<TensorImpl>(a->storage, new_shape, calculate_strides(new_shape), a->requires_grad);
        return make_shared<TensorImpl>(a->storage, new_shape, new_strides, a->offset, a->requires_grad);
    }
    // invalid view, copy and reshape
    // copy to "contiguous" tensor
    shared_ptr<TensorImpl> res = contiguous(a);

    // reshape
    res->shape = new_shape;
    res->strides = calculate_strides(new_shape);
    if (res->grad) {
        res->grad->impl()->shape = res->shape;
        res->grad->impl()->strides = res->strides;
    }

    return res;
}
std::shared_ptr<TensorImpl> slice(std::shared_ptr<const TensorImpl> a, const std::vector<TensorIndex>& indices) {
    if (indices.size() > a->shape.size()) {
        throw std::runtime_error("Too many slice indices");
    }
    std::vector<size_t> new_shape;
    std::vector<size_t> new_strides;
    size_t new_offset = a->offset;
    for (size_t i = 0; i < a->shape.size(); ++i) {
        if (i >= indices.size()) {
            new_shape.push_back(a->shape[i]);
            new_strides.push_back(a->strides[i]);
        } else if (indices[i].type == TensorIndex::Type::Index) {
            new_offset += indices[i].index * a->strides[i];
        } else if (indices[i].type == TensorIndex::Type::Slice) {
            if (indices[i].slice.start >= indices[i].slice.end) {
                throw std::runtime_error("Invalid slice range");
            }
            new_offset += indices[i].slice.start * a->strides[i];
            new_shape.push_back((indices[i].slice.end - indices[i].slice.start)/indices[i].slice.step);
            new_strides.push_back(indices[i].slice.step * a->strides[i]);
        }
    }
    if (new_shape.size() == 0) {
        new_shape.push_back(1);
        new_strides.push_back(1);
    }
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(a->storage, new_shape, new_strides, new_offset, a->requires_grad);
    return res;
}

std::shared_ptr<TensorImpl> contiguous(shared_ptr<const TensorImpl> a) {
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, a->shape, a->requires_grad);

    std::vector<size_t> idx(a->shape.size(), 0);
    const std::vector<size_t>& shape = a->shape;

    size_t a_offset = a->offset;
    size_t res_offset = res->offset;

    const std::vector<size_t>& a_strides = a->strides;
    const std::vector<size_t>& res_strides = res->strides;

    const std::vector<float>& a_data = a->storage->data;
    std::vector<float>& res_data = res->storage->data;

    const size_t n_el = a->n_el;
    for (size_t i = 0; i < n_el; ++i) {
        res_data[res_offset] = a_data[a_offset];
        increment_offset_unary_op(idx, shape, a_offset, a_strides, res_offset, res_strides);
    }
    return res;
}
}
