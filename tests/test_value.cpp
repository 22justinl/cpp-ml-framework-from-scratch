#include "doctest.h"

#include "core/value.h"

TEST_CASE("Value initializes correctly") {
    Value x(1.0);
    CHECK(x.data == doctest::Approx(1.0));
    CHECK(x.grad == doctest::Approx(0.0));
}
