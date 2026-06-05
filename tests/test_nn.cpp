#include "doctest.h"
#include "nn/linear.h"
#include "nn/relu.h"
#include "nn/sequential.h"

TEST_CASE("Test MLP 1") {
    nn::Sequential m({
            std::make_shared<nn::Linear>(5,7),
            std::make_shared<nn::ReLU>(),
            std::make_shared<nn::Linear>(7,2)
            });
    Tensor x1({1,2,3,4,5}, {1, 5});
    Tensor x2({
            1,2,3,4,5,
            -1,-2,-3,-4,-5,
            2,4,6,8,10,
            -2,-4,-6,-8,-10
            }, {4, 5});
    Tensor res1 = m.forward(x1);
    Tensor res2 = m.forward(x2);
}
