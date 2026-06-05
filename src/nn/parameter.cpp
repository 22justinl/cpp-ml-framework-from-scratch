#include "parameter.h"

namespace nn {
Parameter::Parameter(): tensor(Tensor()) {}
Parameter::Parameter(Tensor tensor): tensor(std::move(tensor)) {}
}
