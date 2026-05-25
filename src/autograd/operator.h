#pragma once

#include "core/tensor.h"

class Operator {
public:
    virtual void backward(const Tensor& grad_output);
};
