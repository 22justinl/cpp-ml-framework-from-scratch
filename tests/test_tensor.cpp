#include "doctest.h"

#include "core/tensor.h"

TEST_CASE("Tensor initialization (empty)") {
    Tensor t;
    CHECK(t.data_raw().size() == 0);
    CHECK(t.grad_raw().size() == 0);
    CHECK(t.shape().size() == 0);
    CHECK(t.strides_raw().size() == 0);
}

TEST_CASE("Tensor initialization (values)") {
    Tensor t({
            0,1,
            2,3,
            4,5,
            6,7
            }, {4,2});
    CHECK(t.data_raw().size() == 8);
    CHECK(t.grad_raw().size() == 8);
    CHECK(t.shape().size() == 2);
    CHECK(t.strides_raw().size() == 2);

    CHECK(t.shape()[0] == 4);
    CHECK(t.shape()[1] == 2);
    CHECK(t.strides_raw()[0] == 2);
    CHECK(t.strides_raw()[1] == 1);
    for (size_t i = 0; i < 8; ++i) {
        CHECK(t.data_raw()[i] == i);
        CHECK(t.grad_raw()[i] == 0);
    }
}

TEST_CASE("Tensor initialization (full)") {
    Tensor t(1.5, {5, 3});
    CHECK(t.data_raw().size() == 15);
    CHECK(t.grad_raw().size() == 15);
    CHECK(t.shape().size() == 2);
    CHECK(t.strides_raw().size() == 2);

    CHECK(t.shape()[0] == 5);
    CHECK(t.shape()[1] == 3);
    CHECK(t.strides_raw()[0] == 3);
    CHECK(t.strides_raw()[1] == 1);
    for (size_t i = 0; i < 15; ++i) {
        CHECK(t.data_raw()[i] == 1.5);
        CHECK(t.grad_raw()[i] == 0);
    }
}

TEST_CASE("Tensor indexing") {
    Tensor t({
            0,1,
            2,3,
            4,5,
            6,7
            }, {4,2});
    CHECK(t.at({0, 1}) == 1);
    CHECK(t.at({3, 0}) == 6);
    CHECK(t({0, 0}) == 0);
    CHECK(t({2, 1}) == 5);
    CHECK_THROWS(t({0, 2}));
    CHECK_THROWS(t({4, 0}));
}

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

TEST_CASE("Tensor matmul") {
    Tensor t1({0,1,2,3,4,5,6,7,8,9,10,11}, {3,4});
    Tensor t2({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}, {4,5});
    Tensor t3({1,1,1}, {1,3});
    Tensor res1 = Tensor::matmul(t1,t2);
    Tensor res2 = Tensor::matmul(t3,t1);
    Tensor expected1({70,76,82,88,94,190,212,234,256,278,310,348,386,424,462}, {3,5});
    Tensor expected2({12,15,18,21}, {1,4});
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            CHECK(res1({i,j}) == expected1({i,j}));
        }
    }
    for (size_t i = 0; i < 4; ++i) {
        CHECK(res2({0, i}) == expected2({0, i}));
    }
}
TEST_CASE("Tensor matmul (error cases)") {
    Tensor t1(0.f, {10,20});
    Tensor t2(0.f, {10,30});
    Tensor t3(0.f, {10,20});
    CHECK_THROWS(Tensor::matmul(t1, t2));
    CHECK_THROWS(Tensor::matmul(t2, t1));
    CHECK_THROWS(Tensor::matmul(t1, t3));
    CHECK_THROWS(Tensor::matmul(t3, t1));
}

TEST_CASE("Tensor matvec") {
    Tensor t1({0,1,2,3,4,5,6,7,8,9,10,11}, {3, 4});
    Tensor t2({0,1,2,3}, {4});
    Tensor t3({0,1,2,3}, {4,1});

    Tensor res1 = Tensor::matvec(t1, t2);
    Tensor res2 = Tensor::matvec(t1, t3);
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
    CHECK_THROWS(Tensor::matvec(t1, t2));
    CHECK_THROWS(Tensor::matvec(t1, t3));
    CHECK_THROWS(Tensor::matvec(t2, t2));
    CHECK_THROWS(Tensor::matvec(t2, t3));
    CHECK_THROWS(Tensor::matvec(t3, t2));
}
TEST_CASE("Tensor dot") {
    Tensor t1({1,2,3}, {3});
    Tensor t2({-1,-2,3}, {3});
    Tensor t3({1,2,3}, {3,1});
    Tensor t4({-1,-2,3}, {3,1});
    Tensor t5({1,2,3}, {1,3});
    Tensor t6({-1,-2,3}, {1,3});

    Tensor d1 = Tensor::dot(t1, t2);
    Tensor d2 = Tensor::dot(t3, t4);
    Tensor d3 = Tensor::dot(t5, t6);
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

    CHECK_THROWS(Tensor::dot(t1, t3));
    CHECK_THROWS(Tensor::dot(t1, t5));
    CHECK_THROWS(Tensor::dot(t3, t2));
    CHECK_THROWS(Tensor::dot(t3, t6));
    CHECK_THROWS(Tensor::dot(t6, t1));
    CHECK_THROWS(Tensor::dot(t6, t3));
}

// TEST_CASE("Tensor zero grad") {
//     Tensor t({0,1,2,3,4,5,6,7}, {2,4});
// }
