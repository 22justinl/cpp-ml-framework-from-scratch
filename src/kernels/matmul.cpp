#include "matmul.h"
#include "tensor_ops.h"
#include "utils/tensor_utils.h"

namespace kernels {
constexpr size_t M_c = 128;  // M cache block size
constexpr size_t N_c = 128;  // N cache block size
constexpr size_t K_c = 128;  // K cache block size
constexpr size_t M_r = 32;    // M register block size
constexpr size_t N_r = 32;    // N register block size

// Pack a_t[start_i:k_c][start_j:m_c] into contiguous buffer _a with shape (K_c, M_c) to fit in cache
// a_t_stride = num cols in a_t = num rows in a
// Assuming a_t is contiguous
void pack_a_t(float* a_t, const size_t kc, const size_t mc, const size_t a_t_stride, float _a[K_c*M_c]) {
    size_t p = 0;

    size_t mp = mc/M_r;     // number of FULL panels
    size_t _M_r = mc % M_r; // nonfull panel size
    float* curr_panel = a_t;
    // pack full panels
    for (size_t mm = 0; mm < mp; ++mm) {
        curr_panel = a_t + mm*M_r;
        for (size_t l = 0; l < kc; ++l) {
            for (size_t m = 0; m < M_r; ++m) {
                _a[p++] = curr_panel[m];
            }
            curr_panel += a_t_stride;
        }
    }
    // pack nonfull panel with padding
    if (_M_r != 0 && _M_r < M_r) {
        curr_panel = a_t + mp*M_r;
        for (size_t l = 0; l < kc; ++l) {
            for (size_t m = 0; m < _M_r; ++m) {
                _a[p++] = curr_panel[m];
            }
            for (size_t m = _M_r; m < M_r; ++m) {
                _a[p++] = 0;
            }
            curr_panel += a_t_stride;
        }
    }
}

// Pack b[start_i:k_c][start_j:n_c] into contiguous buffer _b with shape (K_c, N_c) to fit in cache
// b_stride = num rows in b
// Assuming b is contiguous
void pack_b(float* b, const size_t kc, const size_t nc, const size_t b_stride, float _b[K_c*N_c]) {
    size_t p = 0;

    size_t np = nc/N_r;     // number of FULL panels
    size_t _N_r = nc % N_r; // nonfull panel size
    float* curr_panel = b;
    // pack full panels
    for (size_t nn = 0; nn < np; ++nn) {
        curr_panel = b + nn*N_r;
        for (size_t l = 0; l < kc; ++l) {
            for (size_t n = 0; n < N_r; ++n) {
                _b[p++] = curr_panel[n];
            }
            curr_panel += b_stride;
        }
    }
    // pack nonfull panel with padding
    if (_N_r != 0 && _N_r < N_r) {
        curr_panel = b + np*N_r;
        for (size_t l = 0; l < kc; ++l) {
            for (size_t n = 0; n < _N_r; ++n) {
                _b[p++] = curr_panel[n];
            }
            for (size_t n = _N_r; n < N_r; ++n) {
                _b[p++] = 0;
            }
            curr_panel += b_stride;
        }
    }
}

void kernel_default(
        float* _a, float* _b, float* res_data,
        size_t kc, size_t mr, size_t nr, size_t res_stride
        ) {
    float accum[M_r*N_r] = {};
    for (size_t l = 0; l < kc; ++l) {
        for (size_t m = 0; m < mr; ++m) {
            for (size_t n = 0; n < nr; ++n) {
                accum[m*N_r+n] += _a[m] * _b[n];
            }
        }
        _a += M_r;
        _b += N_r;
    }
    for (size_t m = 0; m < mr; ++m) {
        for (size_t n = 0; n < nr; ++n) {
            res_data[m*res_stride + n] += accum[m*N_r+n];
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
            pack_b(b_data + k*K_c*N + j*N_c, kc, nc, N, _b);

            for (size_t i = 0; i < mb; ++i) {
                mc = (i != mb-1 || _M_c == 0) ? M_c : _M_c;
                mp = (mc+M_r-1)/M_r;
                _M_r = mc % M_r;
                pack_a_t(a_t_data+k*K_c*M+i*M_c, kc, mc, M, _a);

                for (size_t nn = 0; nn < np; ++nn) {
                    nr = (nn != np-1 || _N_r == 0) ? N_r : _N_r;
                    for (size_t mm = 0; mm < mp; ++mm) {
                        mr = (mm != mp-1 || _M_r == 0) ? M_r : _M_r;

                        // Array start pointers:
                        // _a+mm*kc*M_r =                   ptr to packed kc*M_r panel mm
                        // _b+nn*kc*N_r =                   ptr to packed kcxN_r panel nn
                        // res_data + (i*M_c+mm*M_r)*N
                        //          +  j*N_c + nn*N_r =     ptr to block (i, j), panel (mm, nn) with strides (N, 1)
                        kernel_default(
                                _a+mm*M_r*kc, _b+nn*N_r*kc, res_data+(i*M_c+mm*M_r)*N + j*N_c + nn*N_r,
                                kc, mr, nr, N);
                    }
                }
            }
        }
    }



    return res;
}

}

