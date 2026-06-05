#include "module.h"

namespace nn {
std::vector<std::shared_ptr<Parameter>> Module::parameters() {
    std::vector<std::shared_ptr<Parameter>> res = parameters_;
    for (std::shared_ptr<Module> m : modules) {
        for (auto it = m->parameters_.begin(); it < m->parameters_.end(); ++it) {
            res.push_back(*it);
        }
    }
    return parameters_;
}

void Module::zero_grad() {
    for (std::shared_ptr<Parameter> p : parameters_) {
        p->tensor.zero_grad();
    }
}
const std::string& Module::name() const {
    return name_;
}

void Module::register_parameter(std::shared_ptr<Parameter> parameter) {
    parameters_.push_back(parameter);
}

void Module::set_name(std::string name) {
    name_ = name;
}
}
