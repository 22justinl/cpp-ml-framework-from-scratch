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


// TEST_CASE("Tensor zero grad") {
//     Tensor t({0,1,2,3,4,5,6,7}, {2,4});
// }
