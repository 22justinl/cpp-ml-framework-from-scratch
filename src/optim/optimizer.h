#pragma once

#include "nn/parameter.h"

class Optimizer {
public:
    virtual void step() = 0;

    void zero_grad();

    std::vector<nn::Parameter*> parameters_;
    float lr_;
};
