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
