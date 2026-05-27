#pragma once

class Operator {
public:
    virtual void backward() = 0;
    ~Operator() = default;
};
