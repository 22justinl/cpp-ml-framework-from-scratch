#include "doctest.h"

#include "utils/tensor_utils.h"

TEST_CASE("Sigmoid") {
    Tensor t1({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23}, {2,2,3,2});
    print_tensor(t1);
}
