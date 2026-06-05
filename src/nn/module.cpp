#include "module.h"

namespace nn {
Tensor Module::forward(Tensor x) {
    throw std::runtime_error("Unsupported");
};
std::vector<Parameter*> Module::parameters() {
    std::vector<Parameter*> res = parameters_;
    for (std::shared_ptr<Module> m : modules) {
        // for (auto it = m->parameters_.begin(); it < m->parameters_.end(); ++it) {
        //     res.push_back(*it);
        // }
        for (Parameter* p : m->parameters()) {
            res.push_back(p);
        }
    }
    return res;
}

void Module::zero_grad() {
    for (Parameter* p : parameters_) {
        p->tensor.zero_grad();
    }
}
const std::string& Module::name() const {
    return name_;
}

void Module::register_parameter(Parameter* parameter) {
    parameters_.push_back(parameter);
}

void Module::set_name(std::string name) {
    name_ = name;
}
}
