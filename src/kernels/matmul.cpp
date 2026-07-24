#include "matmul.h"
#include "tensor_ops.h"
#include "utils/tensor_utils.h"
#include <immintrin.h>

namespace kernels {
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

void kernel_4x4_sse(
        float* _a, float* _b, float* res_data,
        size_t kc, size_t mr, size_t nr, size_t res_stride
        ) {
    __m128 _a00, _a11, _a22, _a33;  // vector for each element of _a
    __m128 _b04;                    // vector for a row of _b
    __m128 _c0, _c1, _c2, _c3;      // vector for each row of _c

    _c0 = _mm_setzero_ps();
    _c1 = _mm_setzero_ps();
    _c2 = _mm_setzero_ps();
    _c3 = _mm_setzero_ps();

    for (size_t l = 0; l < kc; ++l) {
        // load each element of _a into vectors
        _a00 = _mm_load1_ps(_a);
        _a11 = _mm_load1_ps(_a+1);
        _a22 = _mm_load1_ps(_a+2);
        _a33 = _mm_load1_ps(_a+3);

        // load one row of _b
        _b04 = _mm_load_ps(_b);

        // calculate row 0
        _c0 = _mm_add_ps(_c0, _mm_mul_ps(_a00, _b04));    // _c[0:4] += _c[0:4] + _a[0]*_b[0:4]
        // calculate row 1
        _c1 = _mm_add_ps(_c1, _mm_mul_ps(_a11, _b04));
        // calculate row 2
        _c2 = _mm_add_ps(_c2, _mm_mul_ps(_a22, _b04));
        // calculate row 3
        _c3 = _mm_add_ps(_c3, _mm_mul_ps(_a33, _b04));

        // move data pointers to next row/col
        _a += M_r;
        _b += N_r;
    }
    // store rows/cols into res_data
    _mm_store_ps(res_data, _c0);
    _mm_store_ps(res_data+4, _c1);
    _mm_store_ps(res_data+8, _c2);
    _mm_store_ps(res_data+12, _c3);
}

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
    alignas(16) float _a[K_c * M_c];
    alignas(16) float _b[K_c * N_c];

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
    alignas(16) float* res_data = res->storage->data.data();
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
                        // kernel_default(
                        //         _a+mm*M_r*kc, _b+nn*N_r*kc, res_data+(i*M_c+mm*M_r)*N + j*N_c + nn*N_r,
                        //         kc, mr, nr, N);
                        kernel_4x4_sse(
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

