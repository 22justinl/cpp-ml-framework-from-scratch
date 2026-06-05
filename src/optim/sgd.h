#pragma once

#include "optim/optimizer.h"

class SGD: public Optimizer {
public:
    SGD(std::vector<nn::Parameter*> parameters, float lr = 1e-2);
    void step() override;
};
