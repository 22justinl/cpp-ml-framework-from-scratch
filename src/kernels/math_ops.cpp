#include "kernels/math_ops.h"

#include "kernels/tensor_ops.h"
#include "utils/tensor_utils.h"

#include <numbers>

using std::shared_ptr;
using std::make_shared;

namespace kernels {
/*
constexpr size_t M_c = 128;  // M cache block size
constexpr size_t N_c = 128;  // N cache block size
constexpr size_t K_c = 128;  // K cache block size
constexpr size_t M_r = 32;    // M register block size
constexpr size_t N_r = 32;    // N register block size
*/

void add_inplace(shared_ptr<TensorImpl> a, shared_ptr<const TensorImpl> b) {
    elementwise_binary_op_inplace(a, b, [](float a, float b) { return a + b; });
}
void sub_inplace(shared_ptr<TensorImpl> a, shared_ptr<const TensorImpl> b) {
    elementwise_binary_op_inplace(a, b, [](float a, float b) { return a - b; });
}
void add_inplace_broadcast(shared_ptr<TensorImpl> a, shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info) {
    elementwise_binary_op_inplace_broadcast(a, b, [](float a, float b) { return a + b; }, b_info);
}
void sub_inplace_broadcast(shared_ptr<TensorImpl> a, shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info) {
    elementwise_binary_op_inplace_broadcast(a, b, [](float a, float b) { return a - b; }, b_info);
}

void zero_inplace(shared_ptr<TensorImpl> a) {
    std::vector<float>& a_data = a->storage->data;
    for (size_t i = 0; i < a_data.size(); ++i) {
        a_data[i] = 0;
    }
}

shared_ptr<TensorImpl> add(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b) {
    return elementwise_binary_op(a, b, [](float a, float b){ return a + b; });
}
shared_ptr<TensorImpl> sub(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b) {
    return elementwise_binary_op(a, b, [](float a, float b){ return a - b; });
}
shared_ptr<TensorImpl> mul(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b) {
    return elementwise_binary_op(a, b, [](float a, float b){ return a * b; });
}
shared_ptr<TensorImpl> div(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b) {
    return elementwise_binary_op(a, b, [](float a, float b){ return a / b; });
}
shared_ptr<TensorImpl> add_broadcast(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info) {
    return elementwise_binary_op_broadcast(a, b, [](float a, float b){ return a + b; }, b_info);
}
shared_ptr<TensorImpl> sub_broadcast(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info) {
    return elementwise_binary_op_broadcast(a, b, [](float a, float b){ return a - b; }, b_info);
}
shared_ptr<TensorImpl> mul_broadcast(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info) {
    return elementwise_binary_op_broadcast(a, b, [](float a, float b){ return a * b; }, b_info);
}
shared_ptr<TensorImpl> div_broadcast(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b, const BroadcastInfo& b_info) {
    return elementwise_binary_op_broadcast(a, b, [](float a, float b){ return a / b; }, b_info);
}
shared_ptr<TensorImpl> div(float f, shared_ptr<const TensorImpl> a) {
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(f, a->shape, a->strides, a->requires_grad);
    const std::vector<float>& a_data = a->storage->data;
    std::vector<float>& res_data = res->storage->data;

    for (size_t i = 0; i < a_data.size(); ++i) {
        res_data[i] /= a_data[i];
    }
    return res;
}
shared_ptr<TensorImpl> scalar_mul(float a, shared_ptr<const TensorImpl> b) {
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, b->shape, b->strides, b->requires_grad);
    for (size_t i = 0; i < b->storage->data.size(); ++i) {
        res->storage->data[i] = a * b->storage->data[i];
    }
    return res;
}

/*
// Pack a_t[start_i:k_c][start_j:m_c] into contiguous buffer _a with shape (K_c, M_c) to fit in cache
// a_t_stride = num cols in a_t = num rows in a
// Assuming a_t is contiguous
void pack_a_t(float* a_t, const size_t start_i, const size_t start_j, const size_t k_c, const size_t m_c, const size_t a_t_stride, float _a[K_c*M_c]) {
    size_t p = 0;

    for (size_t i = 0; i < k_c; ++i) {
        const size_t row_start = (start_i + i) * a_t_stride + start_j;
        for (size_t j = 0; j < m_c; ++j) {
            _a[p++] = a_t[row_start + j];
        }
    }
}
// pack_a_t(a_t_data[k*K_c*M+i*M_c], kc, mc, M, _a);
void pack_a_t2(float* a_t, const size_t kc, const size_t mc, const size_t a_t_stride, float _a[K_c*M_c]) {
    size_t p = 0;

    size_t mp = (mc+M_r-1)/M_r;
    size_t _M_r = mc % M_r;
    float* curr_panel = a_t;
    for (size_t mm = 0; mm < mp; ++mm) {
        size_t mr = (mm != mp-1 || _M_r == 0) ? M_r : _M_r;
        curr_panel = a_t + mm*M_r;
        for (size_t i = 0; i < kc; ++i) {
            for (size_t j = 0; j < mr; ++j) {
                _a[p++] = curr_panel[j];
            }
            curr_panel += a_t_stride;
        }
    }
}

// Pack b[start_i:k_c][start_j:n_c] into contiguous buffer _b with shape (K_c, N_c) to fit in cache
// b_stride = num rows in b
// Assuming b is contiguous
void pack_b(float* b, const size_t start_i, const size_t start_j, const size_t k_c, const size_t n_c, const size_t b_stride, float _b[K_c*N_c]) {
    size_t p = 0;

    for (size_t i = 0; i < k_c; ++i) {
        const size_t row_start = (start_i + i) * b_stride + start_j;
        for (size_t j = 0; j < n_c; ++j) {
            _b[p++] = b[row_start + j];
        }
    }
}
void pack_b2(float* b, const size_t kc, const size_t nc, const size_t b_stride, float _b[K_c*N_c]) {
    size_t p = 0;

    size_t np = (nc+N_r-1)/N_r;
    size_t _N_r = nc % N_r;
    float* curr_panel = b;
    for (size_t nn = 0; nn < np; ++nn) {
        size_t nr = (nn != np-1 || _N_r == 0) ? N_r : _N_r;
        curr_panel = b + nn*N_r;
        for (size_t i = 0; i < kc; ++i) {
            for (size_t j = 0; j < nr; ++j) {
                _b[p++] = curr_panel[j];
            }
            curr_panel += b_stride;
        }
    }
}

void kernel_default(
        float* _a, float* _b, float* res_data,
        size_t kc,
        size_t mc, size_t nc,
        size_t mr, size_t nr,
        size_t N
        ) {
    float accum[M_r*N_r] = {};
    for (size_t l = 0; l < kc; ++l) {
        for (size_t m = 0; m < mr; ++m) {
            for (size_t n = 0; n < nr; ++n) {
                accum[m*N_r+n] += _a[m] * _b[n];
            }
        }
        _a += mc;
        _b += nc;
    }
    for (size_t m = 0; m < mr; ++m) {
        for (size_t n = 0; n < nr; ++n) {
            res_data[m*N + n] += accum[m*N_r+n];
        }
    }
}
void kernel_default2(
        float* _a, float* _b, float* res_data,
        size_t kc,
        size_t mc, size_t nc,
        size_t mr, size_t nr,
        size_t N
        ) {
    float accum[M_r*N_r] = {};
    for (size_t l = 0; l < kc; ++l) {
        for (size_t m = 0; m < mr; ++m) {
            for (size_t n = 0; n < nr; ++n) {
                accum[m*N_r+n] += _a[m] * _b[n];
            }
        }
        _a += mr;
        _b += nr;
    }
    for (size_t m = 0; m < mr; ++m) {
        for (size_t n = 0; n < nr; ++n) {
            res_data[m*N + n] += accum[m*N_r+n];
        }
    }
}

// void kernel_avx2(
//         const std::vector<float>& _a, const std::vector<float>& _b, std::vector<float>& res_data,
//         size_t kc,
//         size_t mr, size_t nr,
//         size_t mm, size_t nn,
//         size_t mc, size_t nc,
//         size_t i, size_t j,
//         size_t N
//         ) {
//     float accum[M_r][N_r] = {};
//     for (size_t l = 0; l < kc; ++l) {
//         for (size_t m = 0; m < mr; ++m) {
//             for (size_t n = 0; n < nr; ++n) {
//                 accum[m][n] += _a[l*mc + m+mm*M_r] * _b[l*nc + n+nn*N_r];
//             }
//         }
//     }
//     for (size_t m = 0; m < mr; ++m) {
//         for (size_t n = 0; n < nr; ++n) {
//             res_data[(i*M_c+m+mm*M_r)*N + j*N_c + n+nn*N_r] += accum[m][n];
//         }
//     }
// }

shared_ptr<TensorImpl> matmul(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b) {
    if (a->shape.size() != 2 || b->shape.size() != 2) {
        throw std::runtime_error("Matrix multiplication only supported for 2D tensors");
    }
    if (a->shape[a->shape.size()-1] != b->shape[0]) {
        throw std::runtime_error("Cannot matrix multiply tensors with shapes " + shape_to_string(a->shape) + " and " + shape_to_string(b->shape));
    }
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, std::vector<size_t>({a->shape[0], b->shape[1]}), a->requires_grad || b->requires_grad);

    if (b->offset > 0 || !is_contiguous(b)) {
        b = kernels::contiguous(b);
    }
    shared_ptr<const TensorImpl> a_t = kernels::transpose(a, 0, 1);
    if (a_t->offset > 0 || !is_contiguous(a_t)) {
        a_t = kernels::contiguous(a_t);
    }

    const size_t M = a->shape[0]; // output dim 1
    const size_t N = b->shape[1]; // ouput dim 2
    const size_t K = a->shape[1]; // shared dim

    // packed buffers
    float _a[K_c * M_c];
    float _b[K_c * N_c];

    // number of blocks
    const size_t mb = (M+M_c-1)/M_c;
    const size_t nb = (N+N_c-1)/N_c;
    const size_t kb = (K+K_c-1)/K_c;
    // last block size
    size_t _M_c = M%M_c;
    size_t _N_c = N%N_c;
    size_t _K_c = K%K_c;
    // current block size (mc = M_c or _M_c etc.)
    size_t mc, nc, kc;

    // number of panels in cache block
    size_t mp, np;
    // last panel size
    size_t _M_r, _N_r;
    // current panel size (mr = M_r or _M_r etc.)
    size_t mr, nr;

    // tensor data
    float* res_data = res->storage->data.data();
    float* a_t_data = a_t->storage->data.data();
    float* b_data = b->storage->data.data();
    for (size_t k = 0; k < kb; ++k) {
        kc = (k != kb-1 || _K_c == 0) ? K_c : _K_c;
        for (size_t j = 0; j < nb; ++j) {
            nc = (j != nb-1 || _N_c == 0) ? N_c : _N_c;
            np = (nc+N_r-1)/N_r;
            _N_r = nc % N_r;
            pack_b2(b_data + k*K_c*N + j*N_c, kc, nc, N, _b);
            // pack_b(b_data, k*K_c, j*N_c, kc, nc, N, _b);
            for (size_t i = 0; i < mb; ++i) {
                mc = (i != mb-1 || _M_c == 0) ? M_c : _M_c;
                mp = (mc+M_r-1)/M_r;
                _M_r = mc % M_r;
                pack_a_t2(a_t_data+k*K_c*M+i*M_c, kc, mc, M, _a);
                // pack_a_t(a_t_data, k*K_c, i*M_c, kc, mc, M, _a);

                for (size_t nn = 0; nn < np; ++nn) {
                    nr = (nn != np-1 || _N_r == 0) ? N_r : _N_r;
                    for (size_t mm = 0; mm < mp; ++mm) {
                        mr = (mm != mp-1 || _M_r == 0) ? M_r : _M_r;

                        // Array start pointers:
                        // _a+mm*M_r =                      _a[mm][0]
                        // _b+nn*N_r =                      _b[nn][0]
                        // res_data + (i*M_c+mm*M_r)*N
                        //          +  j*N_c + nn*N_r =     res_data[i*M_c+mm*M_r][j*N_c+nn*N_rn]  (block (i, j), panel (mm, nn), strides (N, 1))
                        // kernel_default(
                        //         _a+mm*M_r, _b+nn*N_r, res_data+(i*M_c+mm*M_r)*N + j*N_c + nn*N_r,
                        //         kc,
                        //         mc, nc,
                        //         mr, nr,
                        //         N);
                        kernel_default2(
                                _a+mm*M_r, _b+nn*N_r, res_data+(i*M_c+mm*M_r)*N + j*N_c + nn*N_r,
                                kc,
                                mc, nc,
                                mr, nr,
                                N);
                    }
                }
            }
        }
    }



    return res;
}
*/

shared_ptr<TensorImpl> dot(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b) {
    std::vector<size_t> t1_shape = a->shape;
    std::vector<size_t> t2_shape = b->shape;
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(0, std::vector<size_t>({1}), a->requires_grad);
    if (t1_shape.size() == 0 && t2_shape.size() == 0) {
        return res;
    }
    const std::vector<float>& a_data = a->storage->data;
    const std::vector<float>& b_data = b->storage->data;
    float& res_data = res->storage->data[0];
    // a and b 1D tensors
    if (t1_shape.size() == 1 && t2_shape.size() == 1 && t1_shape[0] == t2_shape[0]) {
        for (size_t i = 0; i < t1_shape[0]; ++i) {
            res_data += a_data[i] * b_data[i];
        }
        return res;
    }
    throw std::runtime_error("Cannot dot product tensors with shapes " + shape_to_string(a->shape) + " and " + shape_to_string(b->shape));
}

shared_ptr<TensorImpl> power(shared_ptr<const TensorImpl> a, float x) {
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(a->storage->data, a->shape, a->strides, a->requires_grad);
    for (size_t i = 0; i < a->storage->data.size(); ++i) {
        res->storage->data[i] = std::pow(res->storage->data[i], x);
    }
    return res;
}
shared_ptr<TensorImpl> exp(shared_ptr<const TensorImpl> a) {
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(a->storage->data, a->shape, a->strides, a->requires_grad);
    for (size_t i = 0; i < a->storage->data.size(); ++i) {
        res->storage->data[i] = std::pow(std::numbers::e_v<float>, res->storage->data[i]);
    }
    return res;
}
shared_ptr<TensorImpl> log(shared_ptr<const TensorImpl> a) {
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(a->storage->data, a->shape, a->strides, a->requires_grad);
    for (size_t i = 0; i < a->storage->data.size(); ++i) {
        res->storage->data[i] = std::log(res->storage->data[i]);
    }
    return res;
}

}

// TODO: replace later with reshaping?
shared_ptr<TensorImpl> col_to_vec(shared_ptr<const TensorImpl> a) {
    if (a->shape.size() != 2 || a->shape[1] != 1) {
        throw std::runtime_error("Only 2D column Tensors can be converted to 1D Tensor");
    }
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(a->storage, a->shape, a->strides, a->requires_grad);
    res->shape = {res->shape[0]};
    res->strides = calculate_strides(res->shape);
    if (res->grad) {
        res->grad->impl()->shape = res->shape;
        res->grad->impl()->strides = res->strides;
    }
    return res;
}
shared_ptr<TensorImpl> vec_to_col(shared_ptr<const TensorImpl> a) {
    if (a->shape.size() != 1) {
        throw std::runtime_error("Only 1D Tensors can be converted to column Tensor");
    }
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(a->storage, a->shape, a->strides, a->requires_grad);
    res->shape = {res->shape[0], 1};
    res->strides = calculate_strides(res->shape);
    if (res->grad) {
        res->grad->impl()->shape = res->shape;
        res->grad->impl()->strides = res->strides;
    }
    return res;
}
