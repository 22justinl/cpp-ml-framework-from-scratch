#include "doctest.h"

#include "core/tensor.h"
#include "ops/math_ops.cpp"
#include "utils/tensor_utils.h"

TEST_CASE("Tensor matmul") {
    Tensor t1({0,1,2,3,4,5,6,7,8,9,10,11}, {3,4});
    Tensor t2({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}, {4,5});
    Tensor t3({1,1,1}, {1,3});
    Tensor res1 = matmul(t1,t2);
    Tensor res2 = matmul(t3,t1);
    Tensor expected1({70,76,82,88,94,190,212,234,256,278,310,348,386,424,462}, {3,5});
    Tensor expected2({12,15,18,21}, {1,4});
    CHECK(check_tensor_shape_match(res1, expected1));
    CHECK(check_tensor_shape_match(res2, expected2));
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

TEST_CASE("Tensor matvec") {
    Tensor t1({0,1,2,3,4,5,6,7,8,9,10,11}, {3, 4});
    Tensor t2({0,1,2,3}, {4});
    Tensor t3({0,1,2,3}, {4,1});

    Tensor res1 = matvec(t1, t2);
    Tensor res2 = matvec(t1, t3);
    Tensor expected({14,38,62}, {3,1});
    for (size_t i = 0; i < 3; ++i) {
        CHECK(res1({i,0}) == expected({i,0}));
        CHECK(res2({i,0}) == expected({i,0}));
    }
}
TEST_CASE("Tensor matvec (error cases)") {
    Tensor t1({0,1,2,3,4,5,6,7,8,9,10,11}, {3, 4});
    Tensor t2({0,1,2,3}, {1,4});
    Tensor t3({0,1,2,3, 4}, {5,1});
    CHECK_THROWS(matvec(t1, t2));
    CHECK_THROWS(matvec(t1, t3));
    CHECK_THROWS(matvec(t2, t2));
    CHECK_THROWS(matvec(t2, t3));
    CHECK_THROWS(matvec(t3, t2));
}

TEST_CASE("Tensor dot") {
    Tensor t1({1,2,3}, {3});
    Tensor t2({-1,-2,3}, {3});
    Tensor t3({1,2,3}, {3,1});
    Tensor t4({-1,-2,3}, {3,1});
    Tensor t5({1,2,3}, {1,3});
    Tensor t6({-1,-2,3}, {1,3});

    Tensor d1 = dot(t1, t2);
    Tensor d2 = dot(t3, t4);
    Tensor d3 = dot(t5, t6);
    CHECK(d1({0}) == 4);
    CHECK(d2({0}) == 4);
    CHECK(d3({0}) == 4);
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

TEST_CASE("Tensor transpose") {
    Tensor t0 = Tensor();
    Tensor t1({0,1,2,3,4},{5});
    Tensor t11({0,1,2,3,4},{5,1});
    Tensor t12({0,1,2,3,4},{1,5});
    Tensor t2({0,1,2,3,4,5,6,7,8,9,10,11}, {3,4});

}
