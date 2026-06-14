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

TEST_CASE("Tensor transpose (nD) autograd") {
    Tensor t1(1, {2,3,4}, true);
    Tensor res1 = transpose(t1, 0, 2);
    t1.zero_grad();
    res1.backward(Tensor({
                1,2,3,4,5,6,
                7,8,9,10,11,12,
                13,14,15,16,17,18,
                19,20,21,22,23,24}, {4,3,2}));
    Tensor expected1({
            1,7,13,19,
            3,9,15,21,
            5,11,17,23,

            2,8,14,20,
            4,10,16,22,
            6,12,18,24
            }, {2,3,4});
    CHECK(check_tensor_equal(t1.grad(), expected1));
}

TEST_CASE("Tensor squeeze (autograd)") {
    Tensor t1(1, {1,2,1,3,1}, true);
    Tensor res1 = squeeze(t1, 0);
    Tensor res2 = squeeze(res1, 1);
    Tensor res3 = squeeze(res2, 2);
    t1.zero_grad();
    res3.backward(Tensor({1,2,3,4,5,6}, {2,3}));
    CHECK(check_tensor_equal(t1.grad(), Tensor({1,2,3,4,5,6}, {1,2,1,3,1})));
}

TEST_CASE("Tensor unsqueeze (autograd)") {
    Tensor t1(1, {2,3}, true);
    Tensor res1 = unsqueeze(t1, 0);
    Tensor res2 = unsqueeze(res1, 2);
    Tensor res3 = unsqueeze(res2, 4);
    t1.zero_grad();
    res3.backward(Tensor({1,2,3,4,5,6}, {1,2,1,3,1}));
    CHECK(check_tensor_equal(t1.grad(), Tensor({1,2,3,4,5,6}, {2,3})));
}

TEST_CASE("Tensor reshape (valid view: no copy) (autograd)") {
    Tensor t1(1, {2,3,4}, true);
    Tensor res1 = reshape(t1, {3,4,2});
    t1.zero_grad();
    res1.backward(Tensor({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, {3,4,2}));
    CHECK(check_tensor_equal(t1.grad(), reshape(Tensor({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, {3,4,2}), {2,3,4})));
}

TEST_CASE("Tensor reshape (invalid view: copy) (autograd)") {
    Tensor t1(1, {2,3,4}, true);
    Tensor res1 = reshape(transpose(t1, 0, 1), {3,4,2});
    t1.zero_grad();
    res1.backward(Tensor({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, {3,4,2}));
    CHECK(check_tensor_equal(t1.grad(), transpose(reshape(Tensor({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, {3,4,2}), {3,2,4}), 0, 1)));
}
