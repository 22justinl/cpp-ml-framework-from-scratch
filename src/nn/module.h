#pragma once

#include "parameter.h"
#include <string>

namespace nn {
class Module {
public:
    virtual Tensor forward(Tensor x) = 0;

    ~Module() = default;

    std::vector<std::shared_ptr<Parameter>> parameters();
    void zero_grad();
    const std::string& name() const;

    std::vector<std::shared_ptr<Module>> modules;
protected:
    void register_parameter(std::shared_ptr<Parameter> parameter);
    void set_name(std::string name);
private:
    std::vector<std::shared_ptr<Parameter>> parameters_;
    std::string name_;
};
}
