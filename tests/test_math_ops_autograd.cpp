#include "doctest.h"

#include "core/tensor.h"
#include "ops/math_ops.h"
#include "utils/tensor_utils.h"

TEST_CASE("Tensor add autograd") {
    Tensor t1({0,1,2,3,4,5,6,7}, {2,4}, true);
    Tensor t2({7,6,5,4,3,2,1,0}, {2,4}, true);
    Tensor t3({0,1,2,3,4,5,6,7}, {2,4}, true);

    Tensor add1 = t1 + t2;
    t1.zero_grad();
    t2.zero_grad();
    t3.zero_grad();
    add1.backward();
    for (size_t i = 0; i < 8; ++i) {
        CHECK(t1.grad().data_raw()[i] == 1.f);
        CHECK(t2.grad().data_raw()[i] == 1.f);
    }
    Tensor add2 = t3 + t1;
    t1.zero_grad();
    add2.backward();
    for (size_t i = 0; i < 8; ++i) {
        CHECK(t1.grad().data_raw()[i] == 1.f);
        CHECK(t3.grad().data_raw()[i] == 1.f);
    }
}

TEST_CASE("Tensor sub autograd") {
    Tensor t1({0,1,2,3,4,5,6,7}, {2,4}, true);
    Tensor t2({7,6,5,4,3,2,1,0}, {2,4}, true);
    Tensor t3({0,1,2,3,4,5,6,7}, {2,4}, true);

    Tensor sub1 = t1 - t2;
    t1.zero_grad();
    t2.zero_grad();
    t3.zero_grad();
    sub1.backward();
    for (size_t i = 0; i < 8; ++i) {
        CHECK(t1.grad().data_raw()[i] == 1.f);
        CHECK(t2.grad().data_raw()[i] == -1.f);
    }
    Tensor sub2 = t3 - t1;
    t1.zero_grad();
    sub2.backward();
    for (size_t i = 0; i < 8; ++i) {
        CHECK(t1.grad().data_raw()[i] == -1.f);
        CHECK(t3.grad().data_raw()[i] == 1.f);
    }
}

TEST_CASE("Tensor mul autograd") {
    Tensor t1({0,1,2,3,4,5,6,7}, {2,4}, true);
    Tensor t2({2,2,2,2,2,2,2,2}, {2,4}, true);

    Tensor mul1 = t1 * t2;
    t1.zero_grad();
    t2.zero_grad();
    mul1.backward();
    for (size_t i = 0; i < 8; ++i) {
        CHECK(t1.grad().data_raw()[i] == t2.data_raw()[i]);
        CHECK(t2.grad().data_raw()[i] == t1.data_raw()[i]);
    }
}

TEST_CASE("Tensor div autograd") {
    Tensor t1({0,2,4,6,8,10,12,14}, {2,4}, true);
    Tensor t2({2,2,2,2,2,2,2,2}, {2,4}, true);

    Tensor div1 = t1 / t2;
    t1.zero_grad();
    t2.zero_grad();
    div1.backward();
    for (size_t i = 0; i < 8; ++i) {
        CHECK(t1.grad().data_raw()[i] == 0.5f);
        CHECK(t2.grad().data_raw()[i] == -t1.data_raw()[i]/(t2.data_raw()[i] * t2.data_raw()[i]));
    }
}

TEST_CASE("Tensor matmul autograd") {
    Tensor t1({0,1,2,3,4,5}, {2, 3}, true);
    Tensor t2({6,7,8,9,10,11,12,13,14,15,16,17}, {3, 4}, true);

    Tensor res = matmul(t1, t2);
    t1.zero_grad();
    t2.zero_grad();
    res.backward();
    Tensor expected1({30,46,62,30,46,62}, {2,3});
    Tensor expected2({3,3,3,3,5,5,5,5,7,7,7,7}, {3,4});
    CHECK(check_tensor_equal(t1.grad(), expected1));
    CHECK(check_tensor_equal(t2.grad(), expected2));
}

TEST_CASE("Tensor matvec autograd") {
    Tensor t1({1,2,3,4,5,6,7,8,9,10,11,12}, {3, 4}, true);
    Tensor t2({10, 20, 30, 40}, {4}, true);

    Tensor res = matvec(t1, t2);
    t1.zero_grad();
    t2.zero_grad();
    res.backward();
    Tensor expected1({10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40}, {3, 4});
    Tensor expected2({15,18,21,24}, {4});
    CHECK(check_tensor_equal(t1.grad(), expected1));
    CHECK(check_tensor_equal(t2.grad(), expected2));
}

TEST_CASE("Tensor dot autograd") {
    Tensor t1({1,2,3,4}, {4}, true);
    Tensor t2({1,2,3,4}, {4}, true);

    Tensor res1 = dot(t1, t2);
    t1.zero_grad();
    t2.zero_grad();
    res1.backward();
    CHECK(check_tensor_equal(t1.grad(), t2));
    CHECK(check_tensor_equal(t2.grad(), t1));
}

