#pragma once

#include "core/tensor.h"

namespace nn {
class Parameter {
public:
    Parameter();
    Parameter(Tensor tensor);
    Tensor tensor;
};
}
