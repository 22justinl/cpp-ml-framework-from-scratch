#include "doctest.h"

#include "core/tensor.h"
#include "ops/math_ops.h"
#include "utils/tensor_utils.h"

TEST_CASE("Tensor add broadcast") {
    Tensor t1({0,1,2,3,4,5,6,7}, {2,4});
    Tensor t2({0,1,2,3}, {1,4});
    Tensor t3({0,1}, {2,1});
    Tensor t4({0,1,2,3}, {4});
    Tensor t5({1}, {1});

    Tensor add1 = t1 + t2;
    Tensor add2 = t1 + t3;
    Tensor add3 = t1 + t4;
    Tensor add4 = t1 + t5;

    Tensor expected1({0,2,4,6,4,6,8,10}, {2,4});
    Tensor expected2({0,1,2,3,5,6,7,8}, {2,4});
    Tensor expected3({0,2,4,6,4,6,8,10}, {2,4});
    Tensor expected4({1,2,3,4,5,6,7,8}, {2,4});

    CHECK(check_tensor_equal(add1, expected1));
    CHECK(check_tensor_equal(add2, expected2));
    CHECK(check_tensor_equal(add3, expected3));
    CHECK(check_tensor_equal(add4, expected4));
}

TEST_CASE("Tensor add broadcast autograd") {
    Tensor t1({0,1,2,3,4,5,6,7}, {2,4}, true);
    Tensor t2({0,1,2,3}, {1,4}, true);
    Tensor t3({0,1}, {2,1}, true);
    Tensor t4({0,1,2,3}, {4}, true);
    Tensor t5({1}, {1}, true);

    Tensor add1 = t1 + t2;
    add1.backward();
    CHECK(check_tensor_equal(t1.grad(), Tensor(1, {2,4})));
    CHECK(check_tensor_equal(t2.grad(), Tensor(2, {1,4})));
    Tensor add2 = t1 + t3;
    t1.zero_grad();
    add2.backward();
    CHECK(check_tensor_equal(t1.grad(), Tensor(1, {2,4})));
    CHECK(check_tensor_equal(t3.grad(), Tensor(4, {2, 1})));
    Tensor add3 = t1 + t4;
    t1.zero_grad();
    add3.backward();
    CHECK(check_tensor_equal(t1.grad(), Tensor(1, {2,4})));
    CHECK(check_tensor_equal(t4.grad(), Tensor(2, {4})));
    Tensor add4 = t1 + t5;
    t1.zero_grad();
    add4.backward();
    CHECK(check_tensor_equal(t1.grad(), Tensor(1, {2,4})));
    CHECK(check_tensor_equal(t5.grad(), Tensor(8, {1})));
}
TEST_CASE("Tensor sub broadcast autograd") {
    Tensor t1({0,1,2,3,4,5,6,7}, {2,4}, true);
    Tensor t2({0,1,2,3}, {1,4}, true);
    Tensor t3({0,1}, {2,1}, true);
    Tensor t4({0,1,2,3}, {4}, true);
    Tensor t5({1}, {1}, true);

    Tensor sub1 = t1 - t2;
    sub1.backward();
    CHECK(check_tensor_equal(t1.grad(), Tensor(1, {2,4})));
    CHECK(check_tensor_equal(t2.grad(), Tensor(-2, {1,4})));
    Tensor sub2 = t1 - t3;
    t1.zero_grad();
    sub2.backward();
    CHECK(check_tensor_equal(t1.grad(), Tensor(1, {2,4})));
    CHECK(check_tensor_equal(t3.grad(), Tensor(-4, {2, 1})));
    Tensor sub3 = t1 - t4;
    t1.zero_grad();
    sub3.backward();
    CHECK(check_tensor_equal(t1.grad(), Tensor(1, {2,4})));
    CHECK(check_tensor_equal(t4.grad(), Tensor(-2, {4})));
    Tensor sub4 = t1 - t5;
    t1.zero_grad();
    sub4.backward();
    CHECK(check_tensor_equal(t1.grad(), Tensor(1, {2,4})));
    CHECK(check_tensor_equal(t5.grad(), Tensor(-8, {1})));
}
TEST_CASE("Tensor mul broadcast autograd") {
    Tensor t1({0,1,2,3,4,5,6,7}, {2,4}, true);
    Tensor t2({0,1,2,3}, {1,4}, true);
    Tensor t3({0,1}, {2,1}, true);
    Tensor t4({0,1,2,3}, {4}, true);
    Tensor t5({1}, {1}, true);

    Tensor mul1 = t1 * t2;
    mul1.backward();
    CHECK(check_tensor_equal(t1.grad(), Tensor({0,1,2,3,0,1,2,3}, {2,4})));
    CHECK(check_tensor_equal(t2.grad(), Tensor({4,6,8,10}, {1,4})));
    Tensor mul2 = t1 * t3;
    t1.zero_grad();
    mul2.backward();
    CHECK(check_tensor_equal(t1.grad(), Tensor({0,0,0,0,1,1,1,1}, {2,4})));
    CHECK(check_tensor_equal(t3.grad(), Tensor({6, 22}, {2, 1})));
    Tensor mul3 = t1 * t4;
    t1.zero_grad();
    mul3.backward();
    CHECK(check_tensor_equal(t1.grad(), Tensor({0,1,2,3,0,1,2,3}, {2,4})));
    CHECK(check_tensor_equal(t4.grad(), Tensor({4,6,8,10}, {4})));
    Tensor mul4 = t1 * t5;
    t1.zero_grad();
    mul4.backward();
    CHECK(check_tensor_equal(t1.grad(), Tensor({1,1,1,1,1,1,1,1}, {2,4})));
    CHECK(check_tensor_equal(t5.grad(), Tensor({28}, {1})));
}
TEST_CASE("Tensor div broadcast autograd") {
    Tensor t1({0,1,2,3,4,5,6,7}, {2,4}, true);
    Tensor t2({1,2,3,4}, {1,4}, true);
    Tensor t3({1,2}, {2,1}, true);
    Tensor t4({1,2,3,4}, {4}, true);
    Tensor t5({1}, {1}, true);

    Tensor div1 = t1 / t2;
    div1.backward();
    CHECK(check_tensor_equal(t1.grad(), Tensor({1.f,1.f/2,1.f/3,1.f/4,1.f,1.f/2,1.f/3,1.f/4}, {2,4})));
    CHECK(check_tensor_equal(t2.grad(), Tensor({-4, -1.5, -0.8888, -0.625}, {1,4}), 1e-4));
    Tensor div2 = t1 / t3;
    t1.zero_grad();
    div2.backward();
    CHECK(check_tensor_equal(t1.grad(), Tensor({1,1,1,1,0.5,0.5,0.5,0.5}, {2,4})));
    CHECK(check_tensor_equal(t3.grad(), Tensor({-6,-5.5}, {2, 1})));
    Tensor div3 = t1 / t4;
    t1.zero_grad();
    div3.backward();
    CHECK(check_tensor_equal(t1.grad(), Tensor({1.f,1.f/2,1.f/3,1.f/4,1.f,1.f/2,1.f/3,1.f/4}, {2,4})));
    CHECK(check_tensor_equal(t4.grad(), Tensor({-4, -1.5, -0.8888, -0.625}, {4}), 1e-4));
    Tensor div4 = t1 / t5;
    t1.zero_grad();
    div4.backward();
    CHECK(check_tensor_equal(t1.grad(), Tensor({1,1,1,1,1,1,1,1}, {2,4})));
    CHECK(check_tensor_equal(t5.grad(), Tensor({-28}, {1})));
}
