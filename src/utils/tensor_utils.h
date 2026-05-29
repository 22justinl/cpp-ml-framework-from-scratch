#pragma once

#include "core/tensor.h"
#include <string>

bool check_tensor_equal(const Tensor& t1, const Tensor& t2);
bool check_tensor_shape_match(const Tensor& t1, const Tensor& t2);
bool check_tensorimpl_shape_match(std::shared_ptr<const TensorImpl> t1, std::shared_ptr<const TensorImpl> t2);

std::string shape_to_string(const std::vector<size_t>& shape);
void print_tensor(const Tensor& t1);

std::vector<size_t> calculate_strides(std::vector<size_t> tensor_shape);
size_t calculate_offset(std::shared_ptr<const TensorImpl> t, const std::vector<size_t>& indices);
std::shared_ptr<TensorImpl> create_tensorimpl(const std::vector<float>& data, const std::vector<size_t>& shape, const std::vector<size_t>& strides, bool requires_grad);
