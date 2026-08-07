#pragma once

#include "core/tensor.h"
#include <string>

bool check_tensor_equal(const Tensor& t1, const Tensor& t2, float eps = 1e-7);
bool check_shape_match(const std::vector<size_t> shape1, const std::vector<size_t> shape2, size_t ignore_suffix_count = 0);

std::string shape_to_string(const std::vector<size_t>& shape);
void print_tensor(const Tensor& t1);
void print_recursive(const Tensor& t1, const std::vector<size_t>& idx, size_t dim);

std::vector<size_t> calculate_strides(const std::vector<size_t>& tensor_shape);
size_t calculate_n_el(const std::vector<size_t>& shape);

size_t idx_to_offset(const std::vector<size_t>& idx, const std::vector<size_t>& shape, const std::vector<size_t>& strides, size_t t_offset);

void increment_offset(std::vector<size_t>& idx, const std::vector<size_t>& shape, size_t& offset, const std::vector<size_t>& strides);
// elementwise ops
void increment_offset_unary_op(std::vector<size_t>& idx, const std::vector<size_t>& shape,
        size_t& a_offset, const std::vector<size_t>& a_strides,
        size_t& res_offset, const std::vector<size_t>& res_strides);
void increment_offset_binary_op(std::vector<size_t>& idx, const std::vector<size_t>& shape,
        size_t& a_offset, const std::vector<size_t>& a_strides,
        size_t& b_offset, const std::vector<size_t>& b_strides,
        size_t& res_offset, const std::vector<size_t>& res_strides);
void increment_offset_matmul_op(std::vector<size_t>& idx, const std::vector<size_t>& shape,
        size_t& a_offset, const std::vector<size_t>& a_strides,
        size_t& b_offset, const std::vector<size_t>& b_strides,
        size_t& c_offset, const std::vector<size_t>& c_strides);


bool is_contiguous(std::shared_ptr<const TensorImpl> a);
