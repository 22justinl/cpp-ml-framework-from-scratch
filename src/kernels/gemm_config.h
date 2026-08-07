#pragma once

#include <cstddef>

constexpr size_t M_c = 512;  // M cache block size
constexpr size_t N_c = 1024;  // N cache block size
constexpr size_t K_c = 256;  // K cache block size


#define KERNEL_4x16_AVX2_FMA
// options:
// - KERNEL_SCALAR,
// - KERNEL_4x4_SSE, KERNEL_4x8_SSE, KERNEL_8x4_SSE,
// - KERNEL_8x8_AVX
// - KERNEL_8x8_AVX2_FMA, KERNEL_16x8_AVX2_FMA, KERNEL_4x16_AVX2_FMA

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
