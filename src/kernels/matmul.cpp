#include "matmul.h"
#include "core/tensor.h"
#include "utils/tensor_utils.h"
#include <immintrin.h>
#include <iostream>

namespace kernels {
// Pack panel from a into contiguous buffer _a with shape (K_c, M_c) contiguously
// row_stride: stride to next row
// col_stride: stride to next col
void pack_a(float* a, const size_t kc, const size_t mc, const size_t row_stride, const size_t col_stride, float _a[K_c*M_c]) {
    size_t p = 0;

    size_t mp = mc/M_r;     // number of FULL panels
    size_t _M_r = mc % M_r; // nonfull panel size
    float* curr_panel = a;
    size_t curr_row = 0;
    // pack full panels
    for (size_t mm = 0; mm < mp; ++mm) {
        curr_panel = a + mm*M_r*row_stride;
        for (size_t l = 0; l < kc; ++l) {
            curr_row = 0;
            for (size_t m = 0; m < M_r; ++m) {
                _a[p++] = curr_panel[curr_row];
                curr_row += row_stride;
            }
            curr_panel += col_stride;
        }
    }
    // pack nonfull panel with padding
    if (_M_r != 0 && _M_r < M_r) {
        curr_panel = a + mp*M_r*row_stride;
        for (size_t l = 0; l < kc; ++l) {
            curr_row = 0;
            for (size_t m = 0; m < _M_r; ++m) {
                _a[p++] = curr_panel[curr_row];
                curr_row += row_stride;
            }
            for (size_t m = _M_r; m < M_r; ++m) {
                _a[p++] = 0;
            }
            curr_panel += col_stride;
        }
    }
}
// Pack panel from a_t into buffer _a with shape (K_c, M_c) contiguously
void pack_a_t(float* a_t, const size_t kc, const size_t mc, const size_t row_stride, const size_t col_stride, float _a[K_c*M_c]) {
    size_t p = 0;

    size_t mp = mc/M_r;     // number of FULL panels
    size_t _M_r = mc % M_r; // nonfull panel size
    float* curr_panel = a_t;
    size_t curr_col = 0;
    // pack full panels
    for (size_t mm = 0; mm < mp; ++mm) {
        curr_panel = a_t + mm*M_r*col_stride;
        for (size_t l = 0; l < kc; ++l) {
            curr_col = 0;
            for (size_t m = 0; m < M_r; ++m) {
                _a[p++] = curr_panel[curr_col];
                curr_col += col_stride;
            }
            curr_panel += row_stride;
        }
    }
    // pack nonfull panel with padding
    if (_M_r != 0 && _M_r < M_r) {
        curr_panel = a_t + mp*M_r*col_stride;
        for (size_t l = 0; l < kc; ++l) {
            curr_col = 0;
            for (size_t m = 0; m < _M_r; ++m) {
                _a[p++] = curr_panel[curr_col];
                curr_col += col_stride;
            }
            for (size_t m = _M_r; m < M_r; ++m) {
                _a[p++] = 0;
            }
            curr_panel += row_stride;
        }
    }
}
// Assuming a_t is contiguous
void pack_a_t_contiguous(float* a_t, const size_t kc, const size_t mc, const size_t row_stride, float _a[K_c*M_c]) {
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
            curr_panel += row_stride;
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
            curr_panel += row_stride;
        }
    }
}

// Pack panel from b into contiguous buffer _b with shape (K_c, N_c) contiguously
// row_stride: stride to next row
// col_stride: stride to next col
void pack_b(float* b, const size_t kc, const size_t nc, const size_t row_stride, const size_t col_stride, float _b[K_c*N_c]) {
    size_t p = 0;

    size_t np = nc/N_r;     // number of FULL panels
    size_t _N_r = nc % N_r; // nonfull panel size
    float* curr_panel = b;
    size_t curr_col = 0;
    // pack full panels
    for (size_t nn = 0; nn < np; ++nn) {
        curr_panel = b + nn*N_r*col_stride;
        for (size_t l = 0; l < kc; ++l) {
            curr_col = 0;
            for (size_t n = 0; n < N_r; ++n) {
                _b[p++] = curr_panel[curr_col];
                curr_col += col_stride;
            }
            curr_panel += row_stride;
        }
    }
    // pack nonfull panel with padding
    if (_N_r != 0 && _N_r < N_r) {
        curr_panel = b + np*N_r*col_stride;
        for (size_t l = 0; l < kc; ++l) {
            curr_col = 0;
            for (size_t n = 0; n < _N_r; ++n) {
                _b[p++] = curr_panel[curr_col];
                curr_col += col_stride;
            }
            for (size_t n = _N_r; n < N_r; ++n) {
                _b[p++] = 0;
            }
            curr_panel += row_stride;
        }
    }
}
// Assuming b is contiguous
void pack_b_contiguous(float* b, const size_t kc, const size_t nc, const size_t row_stride, float _b[K_c*N_c]) {
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
            curr_panel += row_stride;
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
            curr_panel += row_stride;
        }
    }
}
// TODO: update kernels as in kernel_4x16_avx2_fma
// - function signature
// - set c_data buffer directly
// - dont need to consider edge cases
void kernel_default(
        float* _a, float* _b, float* c_rblock,
        size_t kc, size_t mr, size_t nr, size_t c_row_stride
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
            c_rblock[m*c_row_stride + n] += accum[m*N_r+n];
        }
    }
}

void kernel_4x4_sse(
        float* _a, float* _b, float* c_rblock,
        size_t kc, size_t mr, size_t nr, size_t c_row_stride
        ) {
    __m128 _a0, _a1, _a2, _a3;  // vector for each element of _a
    __m128 _b04;                    // vector for a row of _b
    __m128 _c0, _c1, _c2, _c3;      // vector for each row of _c

    _c0 = _mm_setzero_ps();
    _c1 = _mm_setzero_ps();
    _c2 = _mm_setzero_ps();
    _c3 = _mm_setzero_ps();

    for (size_t l = 0; l < kc; ++l) {
        // load each element of _a into vectors
        _a0 = _mm_load1_ps(_a);
        _a1 = _mm_load1_ps(_a+1);
        _a2 = _mm_load1_ps(_a+2);
        _a3 = _mm_load1_ps(_a+3);

        // load one row of _b
        _b04 = _mm_load_ps(_b);

        // calculate row 0
        _c0 = _mm_add_ps(_c0, _mm_mul_ps(_a0, _b04));    // _c[0:4] = _c[0:4] + _a[0]*_b[0:4]
        // calculate row 1
        _c1 = _mm_add_ps(_c1, _mm_mul_ps(_a1, _b04));
        // calculate row 2
        _c2 = _mm_add_ps(_c2, _mm_mul_ps(_a2, _b04));
        // calculate row 3
        _c3 = _mm_add_ps(_c3, _mm_mul_ps(_a3, _b04));

        // move data pointers to next row/col
        _a += M_r;
        _b += N_r;
    }
    // store rows/cols into c_rblock
    if (mr < M_r || nr < N_r) {
        alignas(16) float accum[16];
        _mm_store_ps(accum,    _c0);
        _mm_store_ps(accum+4,  _c1);
        _mm_store_ps(accum+8,  _c2);
        _mm_store_ps(accum+12, _c3);
        for (size_t m = 0; m < mr; ++m) {
            for (size_t n = 0; n < nr; ++n) {
                c_rblock[m*c_row_stride + n] += accum[m*N_r+n];
            }
        }
    } else {
        __m128 c_row;
        c_row = _mm_loadu_ps(c_rblock);
        _mm_storeu_ps(c_rblock, _mm_add_ps(c_row, _c0));
        c_row = _mm_loadu_ps(c_rblock+1*c_row_stride);
        _mm_storeu_ps(c_rblock+1*c_row_stride, _mm_add_ps(c_row, _c1));
        c_row = _mm_loadu_ps(c_rblock+2*c_row_stride);
        _mm_storeu_ps(c_rblock+2*c_row_stride, _mm_add_ps(c_row, _c2));
        c_row = _mm_loadu_ps(c_rblock+3*c_row_stride);
        _mm_storeu_ps(c_rblock+3*c_row_stride, _mm_add_ps(c_row, _c3));
    }
}
void kernel_4x8_sse(
        float* _a, float* _b, float* c_rblock,
        size_t kc, size_t mr, size_t nr, size_t c_row_stride
        ) {
    __m128 _a0, _a1, _a2, _a3;  // vector for each element of _a
    __m128 _b04, _b48;          // vector for a row of _b
    __m128 _c0_04, _c0_48,      // vector for each row of _c
           _c1_04, _c1_48,
           _c2_04, _c2_48,
           _c3_04, _c3_48;

    _c0_04 = _mm_setzero_ps();
    _c0_48 = _mm_setzero_ps();
    _c1_04 = _mm_setzero_ps();
    _c1_48 = _mm_setzero_ps();
    _c2_04 = _mm_setzero_ps();
    _c2_48 = _mm_setzero_ps();
    _c3_04 = _mm_setzero_ps();
    _c3_48 = _mm_setzero_ps();

    for (size_t l = 0; l < kc; ++l) {
        // load each element of _a into vectors
        _a0 = _mm_load1_ps(_a);
        _a1 = _mm_load1_ps(_a+1);
        _a2 = _mm_load1_ps(_a+2);
        _a3 = _mm_load1_ps(_a+3);

        // load one row of _b
        _b04 = _mm_load_ps(_b);
        _b48 = _mm_load_ps(_b+4);

        // calculate row 0
        _c0_04 = _mm_add_ps(_c0_04, _mm_mul_ps(_a0, _b04));    // _c[0:4] = _c[0:4] + _a[0]*_b[0:4]
        _c0_48 = _mm_add_ps(_c0_48, _mm_mul_ps(_a0, _b48));    // _c[4:8] = _c[4:8] + _a[0]*_b[4:8]
        // calculate row 1
        _c1_04 = _mm_add_ps(_c1_04, _mm_mul_ps(_a1, _b04));
        _c1_48 = _mm_add_ps(_c1_48, _mm_mul_ps(_a1, _b48));
        // calculate row 2
        _c2_04 = _mm_add_ps(_c2_04, _mm_mul_ps(_a2, _b04));
        _c2_48 = _mm_add_ps(_c2_48, _mm_mul_ps(_a2, _b48));
        // calculate row 3
        _c3_04 = _mm_add_ps(_c3_04, _mm_mul_ps(_a3, _b04));
        _c3_48 = _mm_add_ps(_c3_48, _mm_mul_ps(_a3, _b48));

        // move data pointers to next row/col
        _a += M_r;
        _b += N_r;
    }
    // store rows/cols into c_rblock
    if (mr < M_r || nr < N_r) {
        alignas(16) float accum[32];
        _mm_store_ps(accum,    _c0_04);
        _mm_store_ps(accum+4,  _c0_48);
        _mm_store_ps(accum+8,  _c1_04);
        _mm_store_ps(accum+12, _c1_48);
        _mm_store_ps(accum+16, _c2_04);
        _mm_store_ps(accum+20, _c2_48);
        _mm_store_ps(accum+24, _c3_04);
        _mm_store_ps(accum+28, _c3_48);
        for (size_t m = 0; m < mr; ++m) {
            for (size_t n = 0; n < nr; ++n) {
                c_rblock[m*c_row_stride + n] += accum[m*N_r+n];
            }
        }
    } else {
        __m128 c_row_04, c_row_48;
        c_row_04 = _mm_loadu_ps(c_rblock);
        _mm_storeu_ps(c_rblock, _mm_add_ps(c_row_04, _c0_04));
        c_row_48 = _mm_loadu_ps(c_rblock+4);
        _mm_storeu_ps(c_rblock+4, _mm_add_ps(c_row_48, _c0_48));

        c_rblock += c_row_stride;
        c_row_04 = _mm_loadu_ps(c_rblock);
        _mm_storeu_ps(c_rblock, _mm_add_ps(c_row_04, _c1_04));
        c_row_48 = _mm_loadu_ps(c_rblock+4);
        _mm_storeu_ps(c_rblock+4, _mm_add_ps(c_row_48, _c1_48));

        c_rblock += c_row_stride;
        c_row_04 = _mm_loadu_ps(c_rblock);
        _mm_storeu_ps(c_rblock, _mm_add_ps(c_row_04, _c2_04));
        c_row_48 = _mm_loadu_ps(c_rblock+4);
        _mm_storeu_ps(c_rblock+4, _mm_add_ps(c_row_48, _c2_48));

        c_rblock += c_row_stride;
        c_row_04 = _mm_loadu_ps(c_rblock);
        _mm_storeu_ps(c_rblock, _mm_add_ps(c_row_04, _c3_04));
        c_row_48 = _mm_loadu_ps(c_rblock+4);
        _mm_storeu_ps(c_rblock+4, _mm_add_ps(c_row_48, _c3_48));
    }
}
void kernel_8x4_sse(
        float* _a, float* _b, float* c_rblock,
        size_t kc, size_t mr, size_t nr, size_t c_row_stride
        ) {
    __m128 _a0, _a1, _a2, _a3,  // vector for each element of _a
           _a4, _a5, _a6, _a7;
    __m128 _b04;                // vector for a row of _b
    __m128 _c0, _c1, _c2, _c3,  // vector for each row of _c
           _c4, _c5, _c6, _c7;

    _c0 = _mm_setzero_ps();
    _c1 = _mm_setzero_ps();
    _c2 = _mm_setzero_ps();
    _c3 = _mm_setzero_ps();
    _c4 = _mm_setzero_ps();
    _c5 = _mm_setzero_ps();
    _c6 = _mm_setzero_ps();
    _c7 = _mm_setzero_ps();

    for (size_t l = 0; l < kc; ++l) {
        // load each element of _a into vectors
        _a0 = _mm_load1_ps(_a);
        _a1 = _mm_load1_ps(_a+1);
        _a2 = _mm_load1_ps(_a+2);
        _a3 = _mm_load1_ps(_a+3);
        _a4 = _mm_load1_ps(_a+4);
        _a5 = _mm_load1_ps(_a+5);
        _a6 = _mm_load1_ps(_a+6);
        _a7 = _mm_load1_ps(_a+7);

        // load one row of _b
        _b04 = _mm_load_ps(_b);

        // calculate row 0
        _c0 = _mm_add_ps(_c0, _mm_mul_ps(_a0, _b04));    // _c[0:4] = _c[0:4] + _a[0]*_b[0:4]
        // calculate row 1
        _c1 = _mm_add_ps(_c1, _mm_mul_ps(_a1, _b04));
        // calculate row 2
        _c2 = _mm_add_ps(_c2, _mm_mul_ps(_a2, _b04));
        // calculate row 3
        _c3 = _mm_add_ps(_c3, _mm_mul_ps(_a3, _b04));
        // calculate row 4
        _c4 = _mm_add_ps(_c4, _mm_mul_ps(_a4, _b04));
        // calculate row 5
        _c5 = _mm_add_ps(_c5, _mm_mul_ps(_a5, _b04));
        // calculate row 6
        _c6 = _mm_add_ps(_c6, _mm_mul_ps(_a6, _b04));
        // calculate row 7
        _c7 = _mm_add_ps(_c7, _mm_mul_ps(_a7, _b04));

        // move data pointers to next row/col
        _a += M_r;
        _b += N_r;
    }
    // store rows/cols into c_rblock
    if (mr < M_r || nr < N_r) {
        alignas(16) float accum[32];
        _mm_store_ps(accum,    _c0);
        _mm_store_ps(accum+4,  _c1);
        _mm_store_ps(accum+8,  _c2);
        _mm_store_ps(accum+12, _c3);
        _mm_store_ps(accum+16, _c4);
        _mm_store_ps(accum+20, _c5);
        _mm_store_ps(accum+24, _c6);
        _mm_store_ps(accum+28, _c7);
        for (size_t m = 0; m < mr; ++m) {
            for (size_t n = 0; n < nr; ++n) {
                c_rblock[m*c_row_stride + n] += accum[m*N_r+n];
            }
        }
    } else {
        __m128 c_row;
        c_row = _mm_loadu_ps(c_rblock);
        _mm_storeu_ps(c_rblock, _mm_add_ps(c_row, _c0));
        c_rblock += c_row_stride;
        c_row = _mm_loadu_ps(c_rblock);
        _mm_storeu_ps(c_rblock, _mm_add_ps(c_row, _c1));
        c_rblock += c_row_stride;
        c_row = _mm_loadu_ps(c_rblock);
        _mm_storeu_ps(c_rblock, _mm_add_ps(c_row, _c2));
        c_rblock += c_row_stride;
        c_row = _mm_loadu_ps(c_rblock);
        _mm_storeu_ps(c_rblock, _mm_add_ps(c_row, _c3));
        c_rblock += c_row_stride;
        c_row = _mm_loadu_ps(c_rblock);
        _mm_storeu_ps(c_rblock, _mm_add_ps(c_row, _c4));
        c_rblock += c_row_stride;
        c_row = _mm_loadu_ps(c_rblock);
        _mm_storeu_ps(c_rblock, _mm_add_ps(c_row, _c5));
        c_rblock += c_row_stride;
        c_row = _mm_loadu_ps(c_rblock);
        _mm_storeu_ps(c_rblock, _mm_add_ps(c_row, _c6));
        c_rblock += c_row_stride;
        c_row = _mm_loadu_ps(c_rblock);
        _mm_storeu_ps(c_rblock, _mm_add_ps(c_row, _c7));

    }
}
void kernel_8x8_avx(
        float* _a, float* _b, float* c_rblock,
        size_t kc, size_t mr, size_t nr, size_t c_row_stride
        ) {
    __m256 _a0, _a1, _a2, _a3,      // vector for each element of _a
           _a4, _a5, _a6, _a7;
    __m256 _b08;                    // vector for a row of _b
    __m256 _c0, _c1, _c2, _c3,      // vector for each row of _c
           _c4, _c5, _c6, _c7;

    _c0 = _mm256_setzero_ps();
    _c1 = _mm256_setzero_ps();
    _c2 = _mm256_setzero_ps();
    _c3 = _mm256_setzero_ps();
    _c4 = _mm256_setzero_ps();
    _c5 = _mm256_setzero_ps();
    _c6 = _mm256_setzero_ps();
    _c7 = _mm256_setzero_ps();

    for (size_t l = 0; l < kc; ++l) {
        // load each element of _a into vectors
        _a0 = _mm256_set1_ps(_a[0]);
        _a1 = _mm256_set1_ps(_a[1]);
        _a2 = _mm256_set1_ps(_a[2]);
        _a3 = _mm256_set1_ps(_a[3]);
        _a4 = _mm256_set1_ps(_a[4]);
        _a5 = _mm256_set1_ps(_a[5]);
        _a6 = _mm256_set1_ps(_a[6]);
        _a7 = _mm256_set1_ps(_a[7]);

        // load one row of _b
        _b08 = _mm256_load_ps(_b);

        // calculate row 0
        _c0 = _mm256_add_ps(_c0, _mm256_mul_ps(_a0, _b08));    // _c[0:8] = _c[0:8] + _a[0]*_b[0:8]
        // calculate row 1
        _c1 = _mm256_add_ps(_c1, _mm256_mul_ps(_a1, _b08));
        // calculate row 2
        _c2 = _mm256_add_ps(_c2, _mm256_mul_ps(_a2, _b08));
        // calculate row 3
        _c3 = _mm256_add_ps(_c3, _mm256_mul_ps(_a3, _b08));
        // calculate row 4
        _c4 = _mm256_add_ps(_c4, _mm256_mul_ps(_a4, _b08));
        // calculate row 5
        _c5 = _mm256_add_ps(_c5, _mm256_mul_ps(_a5, _b08));
        // calculate row 6
        _c6 = _mm256_add_ps(_c6, _mm256_mul_ps(_a6, _b08));
        // calculate row 7
        _c7 = _mm256_add_ps(_c7, _mm256_mul_ps(_a7, _b08));

        // move data pointers to next row/col
        _a += M_r;
        _b += N_r;
    }
    // store rows/cols into c_rblock
    if (mr < M_r || nr < N_r) {
        alignas(32) float accum[64];
        _mm256_store_ps(accum,    _c0);
        _mm256_store_ps(accum+8,  _c1);
        _mm256_store_ps(accum+16, _c2);
        _mm256_store_ps(accum+24, _c3);
        _mm256_store_ps(accum+32, _c4);
        _mm256_store_ps(accum+40, _c5);
        _mm256_store_ps(accum+48, _c6);
        _mm256_store_ps(accum+56, _c7);
        for (size_t m = 0; m < mr; ++m) {
            for (size_t n = 0; n < nr; ++n) {
                c_rblock[m*c_row_stride + n] += accum[m*N_r+n];
            }
        }
    } else {
        __m256 c_row;
        c_row = _mm256_loadu_ps(c_rblock);
        _mm256_storeu_ps(c_rblock, _mm256_add_ps(c_row, _c0));
        c_row = _mm256_loadu_ps(c_rblock+1*c_row_stride);
        _mm256_storeu_ps(c_rblock+1*c_row_stride, _mm256_add_ps(c_row, _c1));
        c_row = _mm256_loadu_ps(c_rblock+2*c_row_stride);
        _mm256_storeu_ps(c_rblock+2*c_row_stride, _mm256_add_ps(c_row, _c2));
        c_row = _mm256_loadu_ps(c_rblock+3*c_row_stride);
        _mm256_storeu_ps(c_rblock+3*c_row_stride, _mm256_add_ps(c_row, _c3));
        c_row = _mm256_loadu_ps(c_rblock+4*c_row_stride);
        _mm256_storeu_ps(c_rblock+4*c_row_stride, _mm256_add_ps(c_row, _c4));
        c_row = _mm256_loadu_ps(c_rblock+5*c_row_stride);
        _mm256_storeu_ps(c_rblock+5*c_row_stride, _mm256_add_ps(c_row, _c5));
        c_row = _mm256_loadu_ps(c_rblock+6*c_row_stride);
        _mm256_storeu_ps(c_rblock+6*c_row_stride, _mm256_add_ps(c_row, _c6));
        c_row = _mm256_loadu_ps(c_rblock+7*c_row_stride);
        _mm256_storeu_ps(c_rblock+7*c_row_stride, _mm256_add_ps(c_row, _c7));
    }
}
void kernel_8x8_avx2_fma(
        float* _a, float* _b, float* c_rblock,
        size_t kc, size_t mr, size_t nr, size_t c_row_stride
        ) {
    __m256 _a0, _a1, _a2, _a3,      // vector for each element of _a
           _a4, _a5, _a6, _a7;
    __m256 _b08;                    // vector for a row of _b
    __m256 _c0, _c1, _c2, _c3,      // vector for each row of _c
           _c4, _c5, _c6, _c7;

    _c0 = _mm256_setzero_ps();
    _c1 = _mm256_setzero_ps();
    _c2 = _mm256_setzero_ps();
    _c3 = _mm256_setzero_ps();
    _c4 = _mm256_setzero_ps();
    _c5 = _mm256_setzero_ps();
    _c6 = _mm256_setzero_ps();
    _c7 = _mm256_setzero_ps();

    for (size_t l = 0; l < kc; ++l) {
        // load each element of _a into vectors
        _a0 = _mm256_set1_ps(_a[0]);
        _a1 = _mm256_set1_ps(_a[1]);
        _a2 = _mm256_set1_ps(_a[2]);
        _a3 = _mm256_set1_ps(_a[3]);
        _a4 = _mm256_set1_ps(_a[4]);
        _a5 = _mm256_set1_ps(_a[5]);
        _a6 = _mm256_set1_ps(_a[6]);
        _a7 = _mm256_set1_ps(_a[7]);

        // load one row of _b
        _b08 = _mm256_load_ps(_b);

        // calculate row 0
        _c0 = _mm256_fmadd_ps(_a0, _b08, _c0);                  // _c[0:8] = _a[0]*_b[0:8] + _c[0:8]
        // calculate row 1
        _c1 = _mm256_fmadd_ps(_a1, _b08, _c1);
        // calculate row 2
        _c2 = _mm256_fmadd_ps(_a2, _b08, _c2);
        // calculate row 3
        _c3 = _mm256_fmadd_ps(_a3, _b08, _c3);
        // calculate row 4
        _c4 = _mm256_fmadd_ps(_a4, _b08, _c4);
        // calculate row 5
        _c5 = _mm256_fmadd_ps(_a5, _b08, _c5);
        // calculate row 6
        _c6 = _mm256_fmadd_ps(_a6, _b08, _c6);
        // calculate row 7
        _c7 = _mm256_fmadd_ps(_a7, _b08, _c7);

        // move data pointers to next row/col
        _a += M_r;
        _b += N_r;
    }
    // store rows/cols into c_rblock
    if (mr < M_r || nr < N_r) {
        alignas(32) float accum[64];
        _mm256_store_ps(accum,    _c0);
        _mm256_store_ps(accum+8,  _c1);
        _mm256_store_ps(accum+16, _c2);
        _mm256_store_ps(accum+24, _c3);
        _mm256_store_ps(accum+32, _c4);
        _mm256_store_ps(accum+40, _c5);
        _mm256_store_ps(accum+48, _c6);
        _mm256_store_ps(accum+56, _c7);
        for (size_t m = 0; m < mr; ++m) {
            for (size_t n = 0; n < nr; ++n) {
                c_rblock[m*c_row_stride + n] += accum[m*N_r+n];
            }
        }
    } else {
        __m256 c_row;
        c_row = _mm256_loadu_ps(c_rblock);
        _mm256_storeu_ps(c_rblock, _mm256_add_ps(c_row, _c0));
        c_row = _mm256_loadu_ps(c_rblock+1*c_row_stride);
        _mm256_storeu_ps(c_rblock+1*c_row_stride, _mm256_add_ps(c_row, _c1));
        c_row = _mm256_loadu_ps(c_rblock+2*c_row_stride);
        _mm256_storeu_ps(c_rblock+2*c_row_stride, _mm256_add_ps(c_row, _c2));
        c_row = _mm256_loadu_ps(c_rblock+3*c_row_stride);
        _mm256_storeu_ps(c_rblock+3*c_row_stride, _mm256_add_ps(c_row, _c3));
        c_row = _mm256_loadu_ps(c_rblock+4*c_row_stride);
        _mm256_storeu_ps(c_rblock+4*c_row_stride, _mm256_add_ps(c_row, _c4));
        c_row = _mm256_loadu_ps(c_rblock+5*c_row_stride);
        _mm256_storeu_ps(c_rblock+5*c_row_stride, _mm256_add_ps(c_row, _c5));
        c_row = _mm256_loadu_ps(c_rblock+6*c_row_stride);
        _mm256_storeu_ps(c_rblock+6*c_row_stride, _mm256_add_ps(c_row, _c6));
        c_row = _mm256_loadu_ps(c_rblock+7*c_row_stride);
        _mm256_storeu_ps(c_rblock+7*c_row_stride, _mm256_add_ps(c_row, _c7));
    }
}
void kernel_16x8_avx2_fma(
        float* _a, float* _b, float* c_rblock,
        size_t kc, size_t mr, size_t nr, size_t c_row_stride
        ) {
    __m256 _a0, _a1, _a2, _a3,      // vector for each element of _a
           _a4, _a5, _a6, _a7,
           _a8, _a9, _a10, _a11,
           _a12, _a13, _a14, _a15;
    __m256 _b08;                    // vector for a row of _b
    __m256 _c0, _c1, _c2, _c3,      // vector for each row of _c
           _c4, _c5, _c6, _c7,
           _c8, _c9, _c10, _c11,
           _c12, _c13, _c14, _c15;

    _c0 = _mm256_setzero_ps();
    _c1 = _mm256_setzero_ps();
    _c2 = _mm256_setzero_ps();
    _c3 = _mm256_setzero_ps();
    _c4 = _mm256_setzero_ps();
    _c5 = _mm256_setzero_ps();
    _c6 = _mm256_setzero_ps();
    _c7 = _mm256_setzero_ps();
    _c8 = _mm256_setzero_ps();
    _c9 = _mm256_setzero_ps();
    _c10 = _mm256_setzero_ps();
    _c11 = _mm256_setzero_ps();
    _c12 = _mm256_setzero_ps();
    _c13 = _mm256_setzero_ps();
    _c14 = _mm256_setzero_ps();
    _c15 = _mm256_setzero_ps();

    for (size_t l = 0; l < kc; ++l) {
        // load each element of _a into vectors
        _a0 = _mm256_set1_ps(_a[0]);
        _a1 = _mm256_set1_ps(_a[1]);
        _a2 = _mm256_set1_ps(_a[2]);
        _a3 = _mm256_set1_ps(_a[3]);
        _a4 = _mm256_set1_ps(_a[4]);
        _a5 = _mm256_set1_ps(_a[5]);
        _a6 = _mm256_set1_ps(_a[6]);
        _a7 = _mm256_set1_ps(_a[7]);
        _a8 = _mm256_set1_ps(_a[8]);
        _a9 = _mm256_set1_ps(_a[9]);
        _a10 = _mm256_set1_ps(_a[10]);
        _a11 = _mm256_set1_ps(_a[11]);
        _a12 = _mm256_set1_ps(_a[12]);
        _a13 = _mm256_set1_ps(_a[13]);
        _a14 = _mm256_set1_ps(_a[14]);
        _a15 = _mm256_set1_ps(_a[15]);

        // load one row of _b
        _b08 = _mm256_load_ps(_b);

        // calculate rows
        _c0 = _mm256_fmadd_ps(_a0, _b08, _c0);                  // _c[0:8] = _a[0]*_b[0:8] + _c[0:8]
        _c1 = _mm256_fmadd_ps(_a1, _b08, _c1);
        _c2 = _mm256_fmadd_ps(_a2, _b08, _c2);
        _c3 = _mm256_fmadd_ps(_a3, _b08, _c3);
        _c4 = _mm256_fmadd_ps(_a4, _b08, _c4);
        _c5 = _mm256_fmadd_ps(_a5, _b08, _c5);
        _c6 = _mm256_fmadd_ps(_a6, _b08, _c6);
        _c7 = _mm256_fmadd_ps(_a7, _b08, _c7);
        _c8 = _mm256_fmadd_ps(_a8, _b08, _c8);
        _c9 = _mm256_fmadd_ps(_a9, _b08, _c9);
        _c10 = _mm256_fmadd_ps(_a10, _b08, _c10);
        _c11 = _mm256_fmadd_ps(_a11, _b08, _c11);
        _c12 = _mm256_fmadd_ps(_a12, _b08, _c12);
        _c13 = _mm256_fmadd_ps(_a13, _b08, _c13);
        _c14 = _mm256_fmadd_ps(_a14, _b08, _c14);
        _c15 = _mm256_fmadd_ps(_a15, _b08, _c15);

        // move data pointers to next row/col
        _a += M_r;
        _b += N_r;
    }
    // store rows/cols into c_rblock
    if (mr < M_r || nr < N_r) {
        alignas(32) float accum[128];
        _mm256_store_ps(accum,    _c0);
        _mm256_store_ps(accum+8,  _c1);
        _mm256_store_ps(accum+8*2, _c2);
        _mm256_store_ps(accum+8*3, _c3);
        _mm256_store_ps(accum+8*4, _c4);
        _mm256_store_ps(accum+8*5, _c5);
        _mm256_store_ps(accum+8*6, _c6);
        _mm256_store_ps(accum+8*7, _c7);
        _mm256_store_ps(accum+8*8, _c8);
        _mm256_store_ps(accum+8*9, _c9);
        _mm256_store_ps(accum+8*10, _c10);
        _mm256_store_ps(accum+8*11, _c11);
        _mm256_store_ps(accum+8*12, _c12);
        _mm256_store_ps(accum+8*13, _c13);
        _mm256_store_ps(accum+8*14, _c14);
        _mm256_store_ps(accum+8*15, _c15);
        for (size_t m = 0; m < mr; ++m) {
            for (size_t n = 0; n < nr; ++n) {
                c_rblock[m*c_row_stride + n] += accum[m*N_r+n];
            }
        }
    } else {
        __m256 c_row;
        c_row = _mm256_loadu_ps(c_rblock);
        _mm256_storeu_ps(c_rblock, _mm256_add_ps(c_row, _c0));
        c_row = _mm256_loadu_ps(c_rblock+1*c_row_stride);
        _mm256_storeu_ps(c_rblock+1*c_row_stride, _mm256_add_ps(c_row, _c1));
        c_row = _mm256_loadu_ps(c_rblock+2*c_row_stride);
        _mm256_storeu_ps(c_rblock+2*c_row_stride, _mm256_add_ps(c_row, _c2));
        c_row = _mm256_loadu_ps(c_rblock+3*c_row_stride);
        _mm256_storeu_ps(c_rblock+3*c_row_stride, _mm256_add_ps(c_row, _c3));
        c_row = _mm256_loadu_ps(c_rblock+4*c_row_stride);
        _mm256_storeu_ps(c_rblock+4*c_row_stride, _mm256_add_ps(c_row, _c4));
        c_row = _mm256_loadu_ps(c_rblock+5*c_row_stride);
        _mm256_storeu_ps(c_rblock+5*c_row_stride, _mm256_add_ps(c_row, _c5));
        c_row = _mm256_loadu_ps(c_rblock+6*c_row_stride);
        _mm256_storeu_ps(c_rblock+6*c_row_stride, _mm256_add_ps(c_row, _c6));
        c_row = _mm256_loadu_ps(c_rblock+7*c_row_stride);
        _mm256_storeu_ps(c_rblock+7*c_row_stride, _mm256_add_ps(c_row, _c7));
        c_row = _mm256_loadu_ps(c_rblock+8*c_row_stride);
        _mm256_storeu_ps(c_rblock+8*c_row_stride, _mm256_add_ps(c_row, _c8));
        c_row = _mm256_loadu_ps(c_rblock+9*c_row_stride);
        _mm256_storeu_ps(c_rblock+9*c_row_stride, _mm256_add_ps(c_row, _c9));
        c_row = _mm256_loadu_ps(c_rblock+10*c_row_stride);
        _mm256_storeu_ps(c_rblock+10*c_row_stride, _mm256_add_ps(c_row, _c10));
        c_row = _mm256_loadu_ps(c_rblock+11*c_row_stride);
        _mm256_storeu_ps(c_rblock+11*c_row_stride, _mm256_add_ps(c_row, _c11));
        c_row = _mm256_loadu_ps(c_rblock+12*c_row_stride);
        _mm256_storeu_ps(c_rblock+12*c_row_stride, _mm256_add_ps(c_row, _c12));
        c_row = _mm256_loadu_ps(c_rblock+13*c_row_stride);
        _mm256_storeu_ps(c_rblock+13*c_row_stride, _mm256_add_ps(c_row, _c13));
        c_row = _mm256_loadu_ps(c_rblock+14*c_row_stride);
        _mm256_storeu_ps(c_rblock+14*c_row_stride, _mm256_add_ps(c_row, _c14));
        c_row = _mm256_loadu_ps(c_rblock+15*c_row_stride);
        _mm256_storeu_ps(c_rblock+15*c_row_stride, _mm256_add_ps(c_row, _c15));
    }
}
void kernel_4x16_avx2_fma(float* _a, float* _b, float* c_rblock, size_t kc) {
    __m256 _a0, _a1, _a2, _a3;      // vector for each element of _a
    __m256 _b08, _b816;             // vector for a row of _b
    __m256 _c0_08, _c0_816, _c1_08, _c1_816,
           _c2_08, _c2_816, _c3_08, _c3_816;      // vector for each row of _c

    _c0_08 = _mm256_setzero_ps();
    _c0_816 = _mm256_setzero_ps();
    _c1_08 = _mm256_setzero_ps();
    _c1_816 = _mm256_setzero_ps();
    _c2_08 = _mm256_setzero_ps();
    _c2_816 = _mm256_setzero_ps();
    _c3_08 = _mm256_setzero_ps();
    _c3_816 = _mm256_setzero_ps();

    for (size_t l = 0; l < kc; ++l) {
        // load each element of _a into vectors
        _a0 = _mm256_set1_ps(_a[0]);
        _a1 = _mm256_set1_ps(_a[1]);
        _a2 = _mm256_set1_ps(_a[2]);
        _a3 = _mm256_set1_ps(_a[3]);

        // load one row of _b
        _b08 = _mm256_load_ps(_b);
        _b816 = _mm256_load_ps(_b+8);

        // calculate row 0
        _c0_08 = _mm256_fmadd_ps(_a0, _b08, _c0_08);                  // _c[0:8] = _a[0]*_b[0:8] + _c[0:8]
        _c0_816 = _mm256_fmadd_ps(_a0, _b816, _c0_816);
        // calculate row 1
        _c1_08 = _mm256_fmadd_ps(_a1, _b08, _c1_08);
        _c1_816 = _mm256_fmadd_ps(_a1, _b816, _c1_816);
        // calculate row 2
        _c2_08 = _mm256_fmadd_ps(_a2, _b08, _c2_08);
        _c2_816 = _mm256_fmadd_ps(_a2, _b816, _c2_816);
        // calculate row 3
        _c3_08 = _mm256_fmadd_ps(_a3, _b08, _c3_08);
        _c3_816 = _mm256_fmadd_ps(_a3, _b816, _c3_816);

        // move data pointers to next row/col
        _a += M_r;
        _b += N_r;
    }
    // store rows/cols into c_rblock
    _mm256_store_ps(c_rblock,    _c0_08);
    _mm256_store_ps(c_rblock+8,  _c0_816);
    _mm256_store_ps(c_rblock+16, _c1_08);
    _mm256_store_ps(c_rblock+24, _c1_816);
    _mm256_store_ps(c_rblock+32, _c2_08);
    _mm256_store_ps(c_rblock+40, _c2_816);
    _mm256_store_ps(c_rblock+48, _c3_08);
    _mm256_store_ps(c_rblock+56, _c3_816);
}

shared_ptr<TensorImpl> matmul(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b) {
    if (a->shape.size() != 2 || b->shape.size() != 2) {
        throw std::runtime_error("Matrix multiplication only supported for 2D tensors");
    }
    if (a->shape[a->shape.size()-1] != b->shape[0]) {
        throw std::runtime_error("Cannot matrix multiply tensors with shapes " + shape_to_string(a->shape) + " and " + shape_to_string(b->shape));
    }
    shared_ptr<TensorImpl> c = make_shared<TensorImpl>(0, std::vector<size_t>({a->shape[0], b->shape[1]}), a->requires_grad || b->requires_grad);
    gemm<GemmCase::MatmulAdd>(1, a, b, 0, c);
    return c;
}

shared_ptr<TensorImpl> scaled_matmul(float alpha, shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b) {
    if (a->shape.size() != 2 || b->shape.size() != 2) {
        throw std::runtime_error("Matrix multiplication only supported for 2D tensors");
    }
    if (a->shape[a->shape.size()-1] != b->shape[0]) {
        throw std::runtime_error("Cannot matrix multiply tensors with shapes " + shape_to_string(a->shape) + " and " + shape_to_string(b->shape));
    }
    shared_ptr<TensorImpl> c = make_shared<TensorImpl>(0, std::vector<size_t>({a->shape[0], b->shape[1]}), a->requires_grad || b->requires_grad);
    gemm<GemmCase::ScaledMatmul>(alpha, a, b, 0, c);
    return c;
}
shared_ptr<TensorImpl> mmadd(shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b, shared_ptr<const TensorImpl> c) {
    if (a->shape.size() != 2 || b->shape.size() != 2) {
        throw std::runtime_error("Matrix multiplication only supported for 2D tensors");
    }
    if (a->shape[a->shape.size()-1] != b->shape[0]) {
        throw std::runtime_error("Cannot matrix multiply tensors with shapes " + shape_to_string(a->shape) + " and " + shape_to_string(b->shape));
    }
    if (a->shape[0] != c->shape[0] || b->shape[1] != c->shape[1]) {
        throw std::runtime_error("Cannot mmadd tensors with shapes " + shape_to_string(a->shape) + ", " + shape_to_string(b->shape) + ", and " + shape_to_string(c->shape));
    }
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(c->storage->data, c->shape, c->strides, a->requires_grad || b->requires_grad || c->requires_grad);
    gemm<GemmCase::MatmulAdd>(1, a, b, 1, res);
    return res;
}
shared_ptr<TensorImpl> mmadd_general(float alpha, shared_ptr<const TensorImpl> a, shared_ptr<const TensorImpl> b, float beta, shared_ptr<const TensorImpl> c) {
    if (a->shape.size() != 2 || b->shape.size() != 2) {
        throw std::runtime_error("Matrix multiplication only supported for 2D tensors");
    }
    if (a->shape[a->shape.size()-1] != b->shape[0]) {
        throw std::runtime_error("Cannot matrix multiply tensors with shapes " + shape_to_string(a->shape) + " and " + shape_to_string(b->shape));
    }
    if (a->shape[0] != c->shape[0] || b->shape[1] != c->shape[1]) {
        throw std::runtime_error("Cannot mmadd tensors with shapes " + shape_to_string(a->shape) + ", " + shape_to_string(b->shape) + ", and " + shape_to_string(c->shape));
    }
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(c->storage->data, c->shape, c->strides, a->requires_grad || b->requires_grad || c->requires_grad);
    gemm<GemmCase::General>(alpha, a, b, beta, res);
    return res;
}
}

