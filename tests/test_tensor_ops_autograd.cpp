#include "doctest.h"

#include "core/tensor.h"
#include "ops/tensor_ops.h"
#include "utils/tensor_utils.h"

TEST_CASE("Tensor transpose autograd") {
    Tensor t1({1,2,3,4,5,6,7,8}, {2,4}, true);
    Tensor res1 = transpose(t1, 0, 1);
    t1.zero_grad();
    res1.backward(Tensor({1,2,3,4,5,6,7,8},{4,2}));
    CHECK(check_tensor_equal(t1.grad(), Tensor({1,3,5,7,2,4,6,8}, {2,4})));
}
