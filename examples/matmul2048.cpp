#include "ops/math_ops.h"
#include "utils/tensor_utils.h"
#include <core/tensor.h>
#include <iostream>

int main() {
    Tensor t1(1, {2048, 2048});
    Tensor t2(2, {2048, 2048});
    Tensor res = matmul(t1, t2);
    std::cout << shape_to_string(res.shape()) << std::endl;
    return 0;
}
