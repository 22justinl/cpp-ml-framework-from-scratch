#pragma once

#include "core/tensor.h"

template <class BinOp>
Tensor binop_helper(const Tensor& t1, const Tensor& t2) {
    Tensor t3 = BinOp::forward(t1, t2);
    if (t1.requires_grad() || t2.requires_grad()) {
        std::shared_ptr<BinOp> op = std::make_shared<BinOp>(t1, t2, t3);
        t3.set_grad_fn(op);
    }
    return t3;
}

template <class MonOp>
Tensor monop_helper(const Tensor& t1) {
    Tensor t2 = MonOp::forward(t1);
    if (t1.requires_grad() || t2.requires_grad()) {
        std::shared_ptr<MonOp> op = std::make_shared<MonOp>(t1, t2);
        t2.set_grad_fn(op);
    }
    return t2;
}

template <class RedOp>
Tensor redop_helper(const Tensor& t1, size_t dim) {
    Tensor t2 = RedOp::forward(t1, dim);
    if (t1.requires_grad() || t2.requires_grad()) {
        std::shared_ptr<RedOp> op = std::make_shared<RedOp>(t1, dim, t2);
        t2.set_grad_fn(op);
    }
    return t2;
}
