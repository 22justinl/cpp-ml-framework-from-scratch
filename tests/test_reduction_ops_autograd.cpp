#include "doctest.h"

#include "core/tensor.h"
#include "ops/reduction_ops.h"
#include "utils/tensor_utils.h"

TEST_CASE("Tensor sum autograd") {
    Tensor t1({1,2,3,4,5,6}, {2,3}, true);
    Tensor res1 = sum(t1);
    t1.zero_grad();
    res1.backward();
    CHECK(check_tensor_equal(t1.grad(), Tensor({1,1,1,1,1,1}, {2,3})));

    // Tensor res2 = sum(t1, 0);
    // t1.zero_grad();
    // res2.backward();
    // print_tensor(t1.grad());
    // CHECK(check_tensor_equal(t1.grad(), Tensor({1,1,1,1,1,1}, {2,3})));
    //
    // Tensor res3 = sum(t1, 1);
    // t1.zero_grad();
    // res3.backward();
    // print_tensor(t1.grad());
    // CHECK(check_tensor_equal(t1.grad(), Tensor({1,1,1,1,1,1}, {2,3})));
}
TEST_CASE("Tensor mean autograd") {
    Tensor t1({1,2,3,4,5,6}, {2,3}, true);
    Tensor res1 = mean(t1);
    t1.zero_grad();
    res1.backward();
    CHECK(check_tensor_equal(t1.grad(), Tensor({1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6}, {2,3})));

    // Tensor res2 = mean(t1, 0);
    // t1.zero_grad();
    // res2.backward();
    // CHECK(check_tensor_equal(t1, Tensor({1.0/2,1.0/2,1.0/2,1.0/2,1.0/2,1.0/2}, {2,3})));
    //
    // Tensor res3 = mean(t1, 1);
    // t1.zero_grad();
    // res3.backward();
    // CHECK(check_tensor_equal(t1, Tensor({1.0/3,1.0/3,1.0/3,1.0/3,1.0/3,1.0/3}, {2,3})));
}
TEST_CASE("Tensor max autograd") {
    Tensor t1({1,2,3,4,5,6}, {2,3}, true);
    Tensor res1 = max(t1);
    t1.zero_grad();
    res1.backward();
    CHECK(check_tensor_equal(t1.grad(), Tensor({0,0,0,0,0,1}, {2,3})));

    // Tensor res2 = max(t1, 0);
    // t1.zero_grad();
    // res2.backward();
    // CHECK(check_tensor_equal(res2.grad(), Tensor({0,0,0,1,1,1}, {2,3})));
    //
    // Tensor res3 = max(t1, 1);
    // t1.zero_grad();
    // res3.backward();
    // CHECK(check_tensor_equal(res3.grad(), Tensor({0,0,1,0,0,1}, {2,3})));

    Tensor t2({1,2,3,4,5,6}, {6}, true);
    Tensor res4 = max(t2);
    t2.zero_grad();
    res4.backward();
    CHECK(check_tensor_equal(t2.grad(), Tensor({0,0,0,0,0,1}, {6})));
}
TEST_CASE("Tensor min autograd") {
    Tensor t1({1,2,3,4,5,6}, {2,3}, true);
    Tensor res1 = min(t1);
    t1.zero_grad();
    res1.backward();
    CHECK(check_tensor_equal(t1.grad(), Tensor({1,0,0,0,0,0}, {2,3})));

    // Tensor res2 = min(t1, 0);
    // t1.zero_grad();
    // res2.backward();
    // CHECK(check_tensor_equal(res2.grad(), Tensor({1,1,1,0,0,0}, {2,3})));
    //
    // Tensor res3 = min(t1, 1);
    // t1.zero_grad();
    // res3.backward();
    // CHECK(check_tensor_equal(res3.grad(), Tensor({1,0,0,1,0,0}, {2,3})));

    Tensor t2({1,2,3,4,5,6}, {6}, true);
    Tensor res4 = min(t2);
    t2.zero_grad();
    res4.backward();
    CHECK(check_tensor_equal(t2.grad(), Tensor({1,0,0,0,0,0}, {6})));
}
