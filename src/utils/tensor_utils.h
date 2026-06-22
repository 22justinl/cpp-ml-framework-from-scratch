#pragma once

#include "core/tensor.h"
#include <string>

void increment_idx(std::shared_ptr<const TensorImpl> t, std::vector<size_t>& idx);
void increment_idx_dim(std::shared_ptr<const TensorImpl> t, std::vector<size_t>& idx, size_t dim);
bool check_tensor_equal(const Tensor& t1, const Tensor& t2, float eps = 1e-7);
bool check_shape_match(const std::vector<size_t> shape1, const std::vector<size_t> shape2);

std::string shape_to_string(const std::vector<size_t>& shape);
void print_tensor(const Tensor& t1);
void print_recursive(const Tensor& t1, const std::vector<size_t>& idx, size_t dim);

std::vector<size_t> calculate_strides(const std::vector<size_t>& tensor_shape);
size_t calculate_n_el(const std::vector<size_t>& shape);

size_t idx_to_offset(const std::vector<size_t>& idx, const std::vector<size_t>& strides, size_t t_offset);
size_t idx_to_offset_checked(const std::vector<size_t>& idx, const std::vector<size_t>& shape, const std::vector<size_t>& strides, size_t t_offset);
