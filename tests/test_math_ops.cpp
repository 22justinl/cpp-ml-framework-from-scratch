#include "doctest.h"

#include "core/tensor.h"
#include "ops/math_ops.h"
#include "utils/tensor_utils.h"

#include <numbers>

TEST_CASE("Tensor add") {
    Tensor t1({0,1,2,3,4,5,6,7}, {2,4});
    Tensor t2({7,6,5,4,3,2,1,0}, {2,4});
    Tensor t3({0,1,2,3,4,5,6,7}, {4,2});

    Tensor add1 = t1 + t2;
    Tensor add2 = t2 + t1;
    for (size_t i = 0; i < 8; ++i) {
        CHECK(add1.data_raw()[i] == 7);
        CHECK(add2.data_raw()[i] == 7);
    }
    CHECK_THROWS(t1+t3);
    CHECK_THROWS(t2+t3);
}

TEST_CASE("Tensor sub") {
    Tensor t1({0,1,2,3,4,5,6,7}, {2,4});
    Tensor t2({0,1,2,3,4,5,6,7}, {2,4});
    Tensor t3({0,1,2,3,4,5,6,7}, {4,2});

    Tensor sub1 = t1 - t2;
    Tensor sub2 = t2 - t1;
    for (size_t i = 0; i < 8; ++i) {
        CHECK(sub1.data_raw()[i] == 0);
        CHECK(sub2.data_raw()[i] == 0);
    }
    CHECK_THROWS(t1-t3);
    CHECK_THROWS(t2-t3);
}

TEST_CASE("Tensor mul") {
    Tensor t1({0,1,2,3,4,5,6,7}, {2,4});
    Tensor t2({2,2,2,2,2,2,2,2}, {2,4});
    Tensor t3({0,1,2,3,4,5,6,7}, {4,2});

    Tensor mul1 = t1 * t2;
    Tensor mul2 = t2 * t1;
    for (size_t i = 0; i < 8; ++i) {
        CHECK(mul1.data_raw()[i] == 2*i);
        CHECK(mul2.data_raw()[i] == 2*i);
    }
    CHECK_THROWS(t1*t3);
    CHECK_THROWS(t2*t3);
}

TEST_CASE("Tensor div") {
    Tensor t1({0,2,4,6,8,10,12,14}, {2,4});
    Tensor t2({2,2,2,2,2,2,2,2}, {2,4});
    Tensor t3({0,1,2,3,4,5,6,7}, {4,2});

    Tensor div1 = t1 / t2;
    for (size_t i = 0; i < 8; ++i) {
        CHECK(div1.data_raw()[i] == i);
    }
    CHECK_THROWS(t2/t1);
    CHECK_THROWS(t1/t3);
    CHECK_THROWS(t2/t3);
}

TEST_CASE("Tensor scalar_mul") {
    float f1 = 2;
    float f2 = -3;
    Tensor t1({1,2,3,4},{4});
    Tensor t2({1,2,3,4,5,6,7,8},{2,4});
    Tensor res1 = scalar_mul(f1, t1);
    Tensor res2 = scalar_mul(t1, f2);
    Tensor res3 = scalar_mul(f2, t2);
    Tensor res4 = scalar_mul(t2, f1);
    CHECK(check_tensor_equal(res1, Tensor({2,4,6,8},{4})));
    CHECK(check_tensor_equal(res2, Tensor({-3,-6,-9,-12},{4})));
    CHECK(check_tensor_equal(res3, Tensor({-3,-6,-9,-12,-15,-18,-21,-24},{2,4})));
    CHECK(check_tensor_equal(res4, Tensor({2,4,6,8,10,12,14,16},{2,4})));
}

TEST_CASE("Tensor matmul") {
    Tensor t1({0,1,2,3,4,5,6,7,8,9,10,11}, {3,4});
    Tensor t2({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}, {4,5});
    Tensor t3({1,1,1}, {1,3});
    Tensor res1 = matmul(t1,t2);
    Tensor res2 = matmul(t3,t1);
    Tensor expected1({70,76,82,88,94,190,212,234,256,278,310,348,386,424,462}, {3,5});
    Tensor expected2({12,15,18,21}, {1,4});
    CHECK(check_tensor_equal(res1, expected1));
    CHECK(check_tensor_equal(res2, expected2));
}
TEST_CASE("Tensor matmul large") {
    Tensor t1(1, {512,1024});
    Tensor t2(2, {1024,1024});
    Tensor res1 = matmul(t1,t2);
    Tensor expected1(2048, {512,1024});
    CHECK(check_tensor_equal(res1, expected1));
}
TEST_CASE("Tensor matmul (error cases)") {
    Tensor t1(0.f, {10,20});
    Tensor t2(0.f, {10,30});
    Tensor t3(0.f, {10,20});
    CHECK_THROWS(matmul(t1, t2));
    CHECK_THROWS(matmul(t2, t1));
    CHECK_THROWS(matmul(t1, t3));
    CHECK_THROWS(matmul(t3, t1));
}

TEST_CASE("Tensor dot") {
    Tensor t1({1,2,3}, {3});
    Tensor t2({-1,-2,3}, {3});
    Tensor d1 = dot(t1, t2);
    CHECK(d1({0}) == 4);
}
TEST_CASE("Tensor dot (error cases)") {
    Tensor t1({1,2,3}, {3});
    Tensor t2({-1,-2,3}, {3});
    Tensor t3({1,2,3}, {3,1});
    Tensor t4({-1,-2,3}, {3,1});
    Tensor t5({1,2,3}, {1,3});
    Tensor t6({-1,-2,3}, {1,3});

    CHECK_THROWS(dot(t1, t3));
    CHECK_THROWS(dot(t1, t5));
    CHECK_THROWS(dot(t3, t2));
    CHECK_THROWS(dot(t3, t6));
    CHECK_THROWS(dot(t6, t1));
    CHECK_THROWS(dot(t6, t3));
}

TEST_CASE("Tensor negative") {
    Tensor t1({1,2,3,-1,-2,0},{2,3});
    Tensor expected1({-1,-2,-3,1,2,0},{2,3});
    Tensor res1 = -t1;
    CHECK(check_tensor_equal(res1, expected1));
}

TEST_CASE("Tensor inplace add") {
    Tensor t1({1,2,3,4,5,6}, {2,3});
    t1 += Tensor({1,-1,2,-2,3,-3}, {2,3});
    CHECK(check_tensor_equal(t1, Tensor({2,1,5,2,8,3}, {2,3})));
}

TEST_CASE("Tensor inplace sub") {
    Tensor t1({1,2,3,4,5,6}, {2,3});
    t1 -= Tensor({1,-1,2,-2,3,-3}, {2,3});
    CHECK(check_tensor_equal(t1, Tensor({0,3,1,6,2,9}, {2,3})));
}

TEST_CASE("Tensor inplace mul") {
    Tensor t1({1,2,3,4,5,6}, {2,3});
    t1 *= Tensor({1,-1,2,-2,3,-3}, {2,3});
    CHECK(check_tensor_equal(t1, Tensor({1,-2,6,-8,15,-18}, {2,3})));
}

TEST_CASE("Tensor inplace div") {
    Tensor t1({1,2,3,4,5,6}, {2,3});
    t1 /= Tensor({1,-1,2,-2,3,-3}, {2,3});
    CHECK(check_tensor_equal(t1, Tensor({1,-2,3.0/2,-2,5.0/3,-2}, {2,3})));
}

TEST_CASE("Tensor power") {
    Tensor t1({0,1,2,3,4,5}, {2,3});
    Tensor t2 = power(t1, 2);
    CHECK(check_tensor_equal(t2, Tensor({0,1,4,9,16,25}, {2,3})));
}

TEST_CASE("Tensor exp") {
    Tensor t1({0, 1, -1}, {3});
    Tensor t2 = exp(t1);
    CHECK(check_tensor_equal(t2, Tensor({1, std::numbers::e_v<float>, 1.0/std::numbers::e_v<float>}, {3})));
}

TEST_CASE("Tensor log") {
    Tensor t1({1, std::numbers::e_v<float>, 1.f/std::numbers::e_v<float>}, {3});
    Tensor t2 = log_e(t1);
    CHECK(check_tensor_equal(t2, Tensor({0, 1.0, -1.0}, {3})));
    // CHECK_THROWS(log(Tensor({0, -1},{2}))); NOTE: Should this throw?
}
