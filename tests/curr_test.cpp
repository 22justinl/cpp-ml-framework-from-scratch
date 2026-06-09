#include "doctest.h"

#include "ops/reduction_ops.h"
#include "utils/tensor_utils.h"

TEST_CASE("Curr test") {
    Tensor t1({
            0,1,2,3,
            4,5,6,7,

            8,9,10,11,
            12,13,14,15,

            16,17,18,19,
            20,21,22,23}, {3,2,4});
    print_tensor(max(t1,0));
    print_tensor(max(t1,1));
    print_tensor(max(t1,2));
}
