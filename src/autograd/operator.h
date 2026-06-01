#pragma once

#include <core/tensor.h>
#include <vector>

class Operator {
public:
    virtual void backward() = 0;
    virtual std::vector<std::shared_ptr<const TensorImpl>> inputs() = 0;
    ~Operator() = default;
};
