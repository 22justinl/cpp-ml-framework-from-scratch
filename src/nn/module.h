#pragma once

#include "parameter.h"
#include <string>

namespace nn {
class Module {
public:
    virtual Tensor forward(Tensor x);

    ~Module() = default;

    std::vector<Parameter*> parameters();
    void zero_grad();
    const std::string& name() const;

    std::vector<std::shared_ptr<Module>> modules;
protected:
    void register_parameter(Parameter* parameter);
    void set_name(std::string name);
private:
    std::vector<Parameter*> parameters_;
    std::string name_;
};
}
