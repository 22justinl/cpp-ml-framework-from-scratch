#include "ops/math_ops.h"
#include "utils/tensor_utils.h"
#include <core/tensor.h>
#include <iostream>

int main() {
    Tensor t1(1, {4096, 4096});
    Tensor t2(2, {4096, 4096});
    Tensor t3(2, {4096, 4096});
    float alpha = 0.5;
    float beta = 0.5;
    Tensor res = mmadd_general(alpha, t1, t2, beta, t3);
    std::cout << shape_to_string(res.shape()) << std::endl;
    return 0;
}
