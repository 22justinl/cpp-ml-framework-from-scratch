#include "sequential.h"

namespace nn {
Sequential::Sequential(std::initializer_list<std::shared_ptr<Module>> sequence, std::string name) {
    set_name(name);
    modules = sequence;
}

Tensor Sequential::forward(Tensor x) {
    for (std::shared_ptr<Module> m_ptr : modules) {
        x = m_ptr->forward(x);
    }
    return x;
}
}
