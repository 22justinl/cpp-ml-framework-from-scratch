#pragma once
#include "core/tensor.h"

using std::shared_ptr;
namespace kernels {
constexpr size_t M_c = 512;  // M cache block size
constexpr size_t N_c = 1024;  // N cache block size
constexpr size_t K_c = 256;  // K cache block size

// Best register block sizes for scalar kernel + auto-vectorization (AVX2, MFA)
// constexpr size_t M_r = 32;    // M register block size
// constexpr size_t N_r = 32;    // N register block size

// 4x4 SSE kernel register block size
// constexpr size_t M_r = 4;
// constexpr size_t N_r = 4;

// 4x8 SSE kernel register block size
// constexpr size_t M_r = 4;
// constexpr size_t N_r = 8;

// 8x4 SSE kernel register block size
// constexpr size_t M_r = 8;
// constexpr size_t N_r = 4;

// 8x8 AVX kernel register block size
// constexpr size_t M_r = 8;
// constexpr size_t N_r = 8;

// 8x8 AVX2 FMA kernel register block size
// constexpr size_t M_r = 8;
// constexpr size_t N_r = 8;

// 16x8 AVX2 FMA kernel register block size
// constexpr size_t M_r = 16;
// constexpr size_t N_r = 8;

// 4x16 AVX2 FMA kernel register block size
constexpr size_t M_r = 4;
constexpr size_t N_r = 16;

void pack_a(float* a, const size_t kc, const size_t mc, const size_t row_stride, const size_t col_stride, float _a[K_c*M_c]);
void pack_a_t(float* a_t, const size_t kc, const size_t mc, const size_t row_stride, const size_t col_stride, float _a[K_c*M_c]);
void pack_a_t_contiguous(float* a_t, const size_t kc, const size_t mc, const size_t row_stride, float _a[K_c*M_c]);
void pack_b(float* b, const size_t kc, const size_t nc, const size_t row_stride, const size_t col_stride, float _b[K_c*N_c]);
void pack_b_contiguous(float* b, const size_t kc, const size_t nc, const size_t row_stride, float _b[K_c*N_c]);

void kernel_default(
        float* _a, float* _b, float* res_data,
        size_t kc, size_t mr, size_t nr, size_t res_stride
        );
void kernel_4x4_sse(
        float* _a, float* _b, float* res_data,
        size_t kc, size_t mr, size_t nr, size_t res_stride
        );
void kernel_4x8_sse(
        float* _a, float* _b, float* res_data,
        size_t kc, size_t mr, size_t nr, size_t res_stride
        );
void kernel_8x4_sse(
        float* _a, float* _b, float* res_data,
        size_t kc, size_t mr, size_t nr, size_t res_stride
        );
void kernel_8x8_avx(
        float* _a, float* _b, float* res_data,
        size_t kc, size_t mr, size_t nr, size_t res_stride
        );
void kernel_8x8_avx2_fma(
        float* _a, float* _b, float* res_data,
        size_t kc, size_t mr, size_t nr, size_t res_stride
        );
void kernel_16x8_avx2_fma(
        float* _a, float* _b, float* res_data,
        size_t kc, size_t mr, size_t nr, size_t res_stride
        );
void kernel_4x16_avx2_fma(
        float* _a, float* _b, float* res_data,
        size_t kc, size_t mr, size_t nr, size_t res_stride
        );

shared_ptr<TensorImpl> matmul(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b);
}
