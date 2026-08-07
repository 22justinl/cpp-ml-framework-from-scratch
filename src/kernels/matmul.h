#pragma once
#include "core/tensor.h"

using std::shared_ptr;
namespace kernels {

enum class GemmCase {
    MatmulAdd,
    ScaledMatmul,
    General
};
enum class MacrokernelCase {
    Add,
    ScaledAdd,
    // BetaAdd, // only needed for alpha = 1, beta != 0
    BetaScaledAdd
};

constexpr size_t M_c = 512;  // M cache block size
constexpr size_t N_c = 1024;  // N cache block size
constexpr size_t K_c = 256;  // K cache block size

#define KERNEL_4x16_AVX2_FMA

#if defined(KERNEL_SCALAR)
#define MICROKERNEL kernel_scalar
// Best register block sizes for scalar kernel + auto-vectorization (AVX2, MFA)
constexpr size_t M_r = 32;    // M register block size
constexpr size_t N_r = 32;    // N register block size
constexpr size_t buffer_alignment = 4;

#elif defined(KERNEL_4x4_SSE)
// 4x4 SSE kernel
#define MICROKERNEL kernel_4x4_sse
constexpr size_t M_r = 4;
constexpr size_t N_r = 4;
constexpr size_t buffer_alignment = 16;

#elif defined(KERNEL_4x8_SSE)
// 4x8 SSE kernel
#define MICROKERNEL kernel_4x8_sse
constexpr size_t M_r = 4;
constexpr size_t N_r = 8;
constexpr size_t buffer_alignment = 16;

#elif defined(KERNEL_8x4_SSE)
// 8x4 SSE kernel
#define MICROKERNEL kernel_8x4_sse
constexpr size_t M_r = 8;
constexpr size_t N_r = 4;
constexpr size_t buffer_alignment = 16;

#elif defined(KERNEL_8x8_AVX)
// 8x8 AVX kernel
#define MICROKERNEL kernel_8x8_avx
constexpr size_t M_r = 8;
constexpr size_t N_r = 8;
constexpr size_t buffer_alignment = 32;

#elif defined(KERNEL_8x8_AVX2_FMA)
// 8x8 AVX2 FMA kernel
#define MICROKERNEL kernel_8x8_avx2_fma
constexpr size_t M_r = 8;
constexpr size_t N_r = 8;
constexpr size_t buffer_alignment = 32;

#elif defined(KERNEL_16x8_AVX2_FMA)
// 16x8 AVX2 FMA kernel
#define MICROKERNEL kernel_16x8_avx2_fma
constexpr size_t M_r = 16;
constexpr size_t N_r = 8;
constexpr size_t buffer_alignment = 32;

#elif defined(KERNEL_4x16_AVX2_FMA)
// 4x16 AVX2 FMA kernel
#define MICROKERNEL kernel_4x16_avx2_fma
constexpr size_t M_r = 4;
constexpr size_t N_r = 16;
constexpr size_t buffer_alignment = 32;
#else
#error "Unknown GEMM Microkernel"
#endif

void pack_a(float* a, const size_t kc, const size_t mc, const size_t row_stride, const size_t col_stride, float _a[K_c*M_c]);
void pack_a_t(float* a_t, const size_t kc, const size_t mc, const size_t row_stride, const size_t col_stride, float _a[K_c*M_c]);
void pack_a_t_contiguous(float* a_t, const size_t kc, const size_t mc, const size_t row_stride, float _a[K_c*M_c]);
void pack_b(float* b, const size_t kc, const size_t nc, const size_t row_stride, const size_t col_stride, float _b[K_c*N_c]);
void pack_b_contiguous(float* b, const size_t kc, const size_t nc, const size_t row_stride, float _b[K_c*N_c]);

void kernel_scalar(
        float* _a, float* _b, float* c_rblock,
        size_t kc, size_t mr, size_t nr);
void kernel_4x4_sse(float* _a, float* _b, float* c_rblock, size_t kc);
void kernel_4x8_sse(float* _a, float* _b, float* c_rblock, size_t kc);
void kernel_8x4_sse(float* _a, float* _b, float* c_rblock, size_t kc);
void kernel_8x8_avx(float* _a, float* _b, float* c_rblock, size_t kc);
void kernel_8x8_avx2_fma(float* _a, float* _b, float* c_rblock, size_t kc);
void kernel_16x8_avx2_fma(float* _a, float* _b, float* c_rblock, size_t kc);
void kernel_4x16_avx2_fma(float* _a, float* _b, float* c_rblock, size_t kc);

shared_ptr<TensorImpl> matmul(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b);
shared_ptr<TensorImpl> scaled_matmul(float alpha, shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b);
shared_ptr<TensorImpl> mmadd(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b, shared_ptr<const TensorImpl> c);
shared_ptr<TensorImpl> mmadd_general(float alpha, shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b, float beta, shared_ptr<const TensorImpl> c);

// process one pair of cache blocks
template<MacrokernelCase Case>
void macrokernel(
        size_t mc, size_t nc, size_t kc,
        float alpha, float beta,
        float* _a, float* _b, float* _c,
        float* c_block, size_t c_row_stride, size_t c_col_stride) {
    // number of register blocks in cache block
    size_t mp = (mc+M_r-1)/M_r;
    size_t np = (nc+N_r-1)/N_r;

    // last register block size
    size_t _M_r = mc % M_r;
    size_t _N_r = nc % N_r;

    // current register block size (mr = M_r or _M_r etc.)
    size_t mr, nr;

    for (size_t nn = 0; nn < np; ++nn) {
        nr = (nn != np-1 || _N_r == 0) ? N_r : _N_r;
        for (size_t mm = 0; mm < mp; ++mm) {
            mr = (mm != mp-1 || _M_r == 0) ? M_r : _M_r;

            float* c_rblock = c_block+(mm*M_r)*c_row_stride + (nn*N_r)*c_col_stride; // ptr to register block

            // Array start pointers:
            // _a+mm*kc*M_r =                   ptr to packed kcxM_r register block mm
            // _b+nn*kc*N_r =                   ptr to packed kcxN_r register block nn
#ifdef KERNEL_SCALAR
            MICROKERNEL(_a+mm*kc*M_r, _b+nn*kc*N_r, _c, kc, mr, nr);
#else
            MICROKERNEL(_a+mm*kc*M_r, _b+nn*kc*N_r, _c, kc);
#endif

            for (size_t m = 0; m < mr; ++m) {
                for (size_t n = 0; n < nr; ++n) {
                    if constexpr (Case == MacrokernelCase::BetaScaledAdd) {
                        // scale c by beta
                        c_rblock[m*c_row_stride + n*c_col_stride] *= beta;
                    }
                    if constexpr (Case == MacrokernelCase::ScaledAdd || Case == MacrokernelCase::BetaScaledAdd) {
                        // add microkernel output scaled by alpha
                        c_rblock[m*c_row_stride + n*c_col_stride] += alpha * _c[m*N_r+n];
                    } else {
                        // add microkernel output
                        c_rblock[m*c_row_stride + n*c_col_stride] += _c[m*N_r+n];
                    }
                }
            }
        }
    }
}

// GEMM implementation (C = alpha * AB + beta * C)
template<GemmCase Case>
void gemm(float alpha, shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b, float beta, shared_ptr<const TensorImpl> c) {
    const size_t M = a->shape[0]; // output dim 1
    const size_t N = b->shape[1]; // ouput dim 2
    const size_t K = a->shape[1]; // shared dim

    // tensor data
    float* a_data = a->storage->data.data() + a->offset;
    float* b_data = b->storage->data.data() + b->offset;
    float* c_data = c->storage->data.data() + c->offset;

    size_t a_row_stride = a->strides[0];
    size_t a_col_stride = a->strides[1];
    size_t b_row_stride = b->strides[0];
    size_t b_col_stride = b->strides[1];
    size_t c_row_stride = c->strides[0];
    size_t c_col_stride = c->strides[1];

    // packed buffers
    alignas(buffer_alignment) float _a[K_c * M_c];
    alignas(buffer_alignment) float _b[K_c * N_c];
    alignas(buffer_alignment) float _c[M_r * N_r];

    // number of cache blocks
    const size_t mb = (M+M_c-1)/M_c;
    const size_t nb = (N+N_c-1)/N_c;
    const size_t kb = (K+K_c-1)/K_c;
    // last cache block size
    size_t _M_c = M%M_c;
    size_t _N_c = N%N_c;
    size_t _K_c = K%K_c;
    // current cache block size (mc = M_c or _M_c etc.)
    size_t mc, nc, kc;
    float* c_block;

    for (size_t j = 0; j < nb; ++j) {
        nc = (j != nb-1 || _N_c == 0) ? N_c : _N_c;
        for (size_t k = 0; k < kb; ++k) {
            kc = (k != kb-1 || _K_c == 0) ? K_c : _K_c;
            pack_b(b_data + (k*K_c)*b_row_stride + (j*N_c)*b_col_stride, kc, nc, b_row_stride, b_col_stride, _b);
            for (size_t i = 0; i < mb; ++i) {
                mc = (i != mb-1 || _M_c == 0) ? M_c : _M_c;
                pack_a(a_data+ (k*K_c)*a_col_stride + (i*M_c)*a_row_stride, kc, mc, a_row_stride, a_col_stride, _a);

                // c_rblock+(i*M_c)*c_row_stride + (j*N_c)*c_col_stride = ptr to cache block (i,j)
                c_block = c_data + (i*M_c)*c_row_stride + (j*N_c)*c_col_stride;
                if constexpr (Case == GemmCase::MatmulAdd) {
                    // C += AB
                    macrokernel<MacrokernelCase::Add>(
                            mc, nc, kc,
                            alpha, beta,
                            _a, _b, _c,
                            c_block, c_row_stride, c_col_stride);
                } else if constexpr (Case == GemmCase::ScaledMatmul) {
                    // C += alpha*AB
                    macrokernel<MacrokernelCase::ScaledAdd>(
                            mc, nc, kc,
                            alpha, beta,
                            _a, _b, _c,
                            c_block, c_row_stride, c_col_stride);
                } else if constexpr (Case == GemmCase::General) {
                    // General case
                    if (k == 0) {
                        // C = alpha*AB + beta*C (only once per (i,j))
                        macrokernel<MacrokernelCase::BetaScaledAdd>(
                                mc, nc, kc,
                                alpha, beta,
                                _a, _b, _c,
                                c_block, c_row_stride, c_col_stride);
                    } else {
                        // C += alpha*AB
                        macrokernel<MacrokernelCase::ScaledAdd>(
                                mc, nc, kc,
                                alpha, 1,
                                _a, _b, _c,
                                c_block, c_row_stride, c_col_stride);
                    }
                }
            }
        }
    }
}


}
