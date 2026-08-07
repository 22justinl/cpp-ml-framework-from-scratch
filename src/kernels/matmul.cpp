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

void kernel_scalar(
        float* _a, float* _b, float* c_rblock,
        size_t kc, size_t mr, size_t nr) {
    for (size_t m = 0; m < mr; ++m) {
        for (size_t n = 0; n < nr; ++n) {
            c_rblock[m*N_r+n] = 0;
        }
    }
    for (size_t l = 0; l < kc; ++l) {
        for (size_t m = 0; m < mr; ++m) {
            for (size_t n = 0; n < nr; ++n) {
                c_rblock[m*N_r+n] += _a[m] * _b[n];
            }
        }
        _a += M_r;
        _b += N_r;
    }
}

void kernel_4x4_sse(float* _a, float* _b, float* c_rblock, size_t kc) {
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
    _mm_store_ps(c_rblock,    _c0);
    _mm_store_ps(c_rblock+4,  _c1);
    _mm_store_ps(c_rblock+8,  _c2);
    _mm_store_ps(c_rblock+12, _c3);
}

void kernel_4x8_sse(float* _a, float* _b, float* c_rblock, size_t kc) {
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
    _mm_store_ps(c_rblock,    _c0_04);
    _mm_store_ps(c_rblock+4,  _c0_48);
    _mm_store_ps(c_rblock+8,  _c1_04);
    _mm_store_ps(c_rblock+12, _c1_48);
    _mm_store_ps(c_rblock+16, _c2_04);
    _mm_store_ps(c_rblock+20, _c2_48);
    _mm_store_ps(c_rblock+24, _c3_04);
    _mm_store_ps(c_rblock+28, _c3_48);
}

void kernel_8x4_sse(float* _a, float* _b, float* c_rblock, size_t kc) {
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
    _mm_store_ps(c_rblock,    _c0);
    _mm_store_ps(c_rblock+4,  _c1);
    _mm_store_ps(c_rblock+8,  _c2);
    _mm_store_ps(c_rblock+12, _c3);
    _mm_store_ps(c_rblock+16, _c4);
    _mm_store_ps(c_rblock+20, _c5);
    _mm_store_ps(c_rblock+24, _c6);
    _mm_store_ps(c_rblock+28, _c7);
}

void kernel_8x8_avx(float* _a, float* _b, float* c_rblock, size_t kc) {
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
    _mm256_store_ps(c_rblock,    _c0);
    _mm256_store_ps(c_rblock+8,  _c1);
    _mm256_store_ps(c_rblock+16, _c2);
    _mm256_store_ps(c_rblock+24, _c3);
    _mm256_store_ps(c_rblock+32, _c4);
    _mm256_store_ps(c_rblock+40, _c5);
    _mm256_store_ps(c_rblock+48, _c6);
    _mm256_store_ps(c_rblock+56, _c7);
}

void kernel_8x8_avx2_fma(float* _a, float* _b, float* c_rblock, size_t kc) {
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
    _mm256_store_ps(c_rblock,    _c0);
    _mm256_store_ps(c_rblock+8,  _c1);
    _mm256_store_ps(c_rblock+16, _c2);
    _mm256_store_ps(c_rblock+24, _c3);
    _mm256_store_ps(c_rblock+32, _c4);
    _mm256_store_ps(c_rblock+40, _c5);
    _mm256_store_ps(c_rblock+48, _c6);
    _mm256_store_ps(c_rblock+56, _c7);
}

void kernel_16x8_avx2_fma(float* _a, float* _b, float* c_rblock, size_t kc) {
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
    _mm256_store_ps(c_rblock,    _c0);
    _mm256_store_ps(c_rblock+8,  _c1);
    _mm256_store_ps(c_rblock+8*2, _c2);
    _mm256_store_ps(c_rblock+8*3, _c3);
    _mm256_store_ps(c_rblock+8*4, _c4);
    _mm256_store_ps(c_rblock+8*5, _c5);
    _mm256_store_ps(c_rblock+8*6, _c6);
    _mm256_store_ps(c_rblock+8*7, _c7);
    _mm256_store_ps(c_rblock+8*8, _c8);
    _mm256_store_ps(c_rblock+8*9, _c9);
    _mm256_store_ps(c_rblock+8*10, _c10);
    _mm256_store_ps(c_rblock+8*11, _c11);
    _mm256_store_ps(c_rblock+8*12, _c12);
    _mm256_store_ps(c_rblock+8*13, _c13);
    _mm256_store_ps(c_rblock+8*14, _c14);
    _mm256_store_ps(c_rblock+8*15, _c15);
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
    size_t n = a->shape.size();
    if (n != b->shape.size()) {
        throw std::runtime_error("Matrix multiplication (non-broadcast) needs same number of dimensions");
    }
    if (a->shape[n-1] != b->shape[n-2] || !check_shape_match(a->shape, b->shape, 2)) {
        throw std::runtime_error("Cannot matrix multiply (non-broadcast) tensors with shapes " + shape_to_string(a->shape) + " and " + shape_to_string(b->shape));
    }
    if (a->shape.size() == 2) {
        shared_ptr<TensorImpl> c = make_shared<TensorImpl>(0, std::vector<size_t>({a->shape[0], b->shape[1]}), a->requires_grad || b->requires_grad);
        gemm<GemmCase::MatmulAdd>(1, a, b, 0, c);
        return c;
    }
    size_t M = a->shape[n-2];
    size_t K = a->shape[n-1];
    size_t N = b->shape[n-1];
    std::vector<size_t> c_shape = a->shape;
    c_shape[n-1] = N;
    shared_ptr<TensorImpl> c = make_shared<TensorImpl>(0, c_shape, a->requires_grad || b->requires_grad);
    std::vector<size_t> idx(n-2, 0);
    shared_ptr<TensorImpl> a_mat = make_shared<TensorImpl>(a->storage, std::vector<size_t>{M, K}, std::vector<size_t>{a->strides[n-2], a->strides[n-1]}, a->offset, a->requires_grad);
    shared_ptr<TensorImpl> b_mat = make_shared<TensorImpl>(b->storage, std::vector<size_t>{K, N}, std::vector<size_t>{b->strides[n-2], b->strides[n-1]}, b->offset, b->requires_grad);
    shared_ptr<TensorImpl> c_mat = make_shared<TensorImpl>(c->storage, std::vector<size_t>{M, N}, std::vector<size_t>{N, 1}, c->offset, c->requires_grad);
    size_t& a_offset = a_mat->offset;
    size_t& b_offset = b_mat->offset;
    size_t& c_offset = c_mat->offset;
    size_t n_mat = calculate_n_el(c->shape)/(M*N);
    for (size_t i = 0; i < n_mat; ++i) {
        gemm<GemmCase::MatmulAdd>(1, a_mat, b_mat, 0, c_mat);
        increment_offset_matmul_op(idx, c->shape, a_offset, a->strides, b_offset, b->strides, c_offset, c->strides);
    }
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
    size_t n = a->shape.size();
    if (n != b->shape.size() || n != c->shape.size()) {
        throw std::runtime_error("MMAdd (non-broadcast) needs same number of dimensions");
    }
    if (
            a->shape[n-1] != b->shape[n-2] ||               // same shared dimension
            a->shape[n-2] != c->shape[n-2] ||               // correct matrix dims for c
            b->shape[n-1] != c->shape[n-1] ||
            !check_shape_match(a->shape, b->shape, 2) ||    // correct prefix dims
            !check_shape_match(a->shape, c->shape, 2)
        ) {
        throw std::runtime_error("Cannot mmadd tensors with shapes " + shape_to_string(a->shape) + ", " + shape_to_string(b->shape) + ", and " + shape_to_string(c->shape));
    }
    if (n == 2) {
        shared_ptr<TensorImpl> res = make_shared<TensorImpl>(c->storage->data, c->shape, c->strides, a->requires_grad || b->requires_grad || c->requires_grad);
        gemm<GemmCase::MatmulAdd>(1, a, b, 1, res);
        return res;
    }
    size_t M = a->shape[n-2];
    size_t K = a->shape[n-1];
    size_t N = b->shape[n-1];
    shared_ptr<TensorImpl> res = make_shared<TensorImpl>(c->storage->data, c->shape, c->strides, a->requires_grad || b->requires_grad || c->requires_grad);
    std::vector<size_t> idx(n-2, 0);
    shared_ptr<TensorImpl> a_mat = make_shared<TensorImpl>(a->storage, std::vector<size_t>{M, K}, std::vector<size_t>{a->strides[n-2], a->strides[n-1]}, a->offset, a->requires_grad);
    shared_ptr<TensorImpl> b_mat = make_shared<TensorImpl>(b->storage, std::vector<size_t>{K, N}, std::vector<size_t>{b->strides[n-2], b->strides[n-1]}, b->offset, b->requires_grad);
    shared_ptr<TensorImpl> res_mat = make_shared<TensorImpl>(res->storage, std::vector<size_t>{M, N}, std::vector<size_t>{res->strides[n-2], res->strides[n-1]}, res->offset, res->requires_grad);
    size_t& a_offset = a_mat->offset;
    size_t& b_offset = b_mat->offset;
    size_t& res_offset = res_mat->offset;
    size_t n_mat = calculate_n_el(res->shape)/(M*N);
    for (size_t i = 0; i < n_mat; ++i) {
        gemm<GemmCase::MatmulAdd>(1, a_mat, b_mat, 1, res_mat);
        increment_offset_matmul_op(idx, res->shape, a_offset, a->strides, b_offset, b->strides, res_offset, c->strides);
    }
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

